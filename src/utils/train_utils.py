import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.functional.segmentation import generalized_dice_score as dice
from pathlib import Path
from monai.losses import DiceCELoss
import pytorch_lightning as pl
from torch.optim import AdamW
import monai.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer


def seed_set(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class HardUnetTrainer(pl.LightningModule):
    def __init__(
        self,
        model,
        loss=DiceCELoss,
        optim=AdamW,
        sched=CosineAnnealingLR,
        lr=0.0001,
        decay=0.01,
        momentum=0.9,
        device="cpu",
        model_type="2D",
        roi_size_w=128,
        roi_size_h=128,
    ):
        super().__init__()
        self.net = model
        self.loss = loss()
        self.dice_metric1 = DiceMetric(reduction="mean_batch", get_not_nans=True)
        self.dice_metric2 = DiceMetric(reduction="mean_batch", get_not_nans=True)
        self.max_epochs = 500
        self.post1 = transforms.Compose([transforms.Activations(sigmoid=True)])
        self.post2 = transforms.Compose([transforms.AsDiscrete(threshold=0.5)])

        if model_type == "3D":
            self.inferer = SlidingWindowInferer(
                roi_size=(roi_size_w, roi_size_h, 64), sw_batch_size=1, overlap=0.25
            )
        else:
            self.inferer = SlidingWindowInferer(
                roi_size=(roi_size_w, roi_size_h), sw_batch_size=1, overlap=0.25
            )
        self.optim = optim(
            self.net.parameters(), lr=lr, weight_decay=decay, momentum=momentum
        )
        self.sched = sched(self.optim, T_max=self.max_epochs)
        self.save_hyperparameters(ignore=["unet", "loss"])

    def num_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        optimizer = self.optim
        scheduler = self.sched
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"].float()
        y_hat = self.net(x)
        y_hat = self.post1(y_hat)
        y_hat = self.post2(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"].float()
        y_hat = self.net(x)
        y_hat = self.post1(y_hat)
        loss = self.loss(y_hat, y)
        y_hat = self.post2(y_hat)
        train_dice = self.dice_metric2(y_hat, y)
        mean_train_dice, _ = self.dice_metric2.aggregate()
        self.log("mean_train_dice", mean_train_dice, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.dice_metric2.reset()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"].float()
        y_hat = self.predict_step(batch, batch_idx)
        val_dice = self.dice_metric1(y_hat, y)
        return {"val_dice": val_dice}

    def on_validation_epoch_end(self):
        mean_val_dice, _ = self.dice_metric1.aggregate()
        self.log("val_dice", mean_val_dice, prog_bar=True)
        self.dice_metric1.reset()


CHANNELS_DIMENSION = 1


def hardunet_train_loop(
    model: nn.Module,
    optim: optim,
    loss: F,
    device: torch.device,
    train_data: DataLoader,
    eval_data: DataLoader = None,
    epochs=1001,
    scheduler: torch.optim.lr_scheduler = None,
    checks=100,
    save=True,
    name="model0",
):  # pragma: no cover
    model = model.to(device)
    n_classes = model.get_classes()
    loss_fn = loss()

    for epoch in range(epochs):
        model.train()
        for batch in train_data:
            batch_X, batch_y = (
                batch["image"].to(device).float(),
                batch["mask"].to(device).float(),
            )
            logits = model(batch_X)
            # y_pred = F.softmax(logits, dim=n_classes)
            loss = loss_fn(logits, batch_y)
            train_dice = sum(
                dice(F.softmax(logits, dim=n_classes), batch_y, n_classes)
            ) / len(batch_y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        if scheduler is not None:
            scheduler.step()

        # Validation phase
        if eval_data is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in eval_data:
                    val_X, val_y = (
                        batch["image"].to(device).float(),
                        batch["mask"].to(device).float(),
                    )
                    logits = model(val_X)
                    # val_pred = F.softmax(logits, dim=n_classes)
                    val_loss = loss_fn(logits, val_y)
                    val_dice = sum(dice(logits, val_y, n_classes)) / len(val_y)
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)

        if (epoch) % checks == 0:
            print(f"Epoch: {epoch}")
            print(f"TRAIN => Loss: {loss.item():.4f} | Average Dice: {train_dice:.4f}")
            if eval_data is not None:
                print(
                    f"EVAL => Loss: {val_loss.item():.4f} | Average Dice: {val_dice:.4f}"
                )
                print(f"Average Val Loss: {avg_val_loss}")
            print("----------------------------------------------------")

    if save:
        save_model(model=model, name=name)


def hardunet_test(
    model: nn.Module,
    device: torch.device,
    test_data: DataLoader,
    loss: F,
    threshold: float = 0.5,
):  # pragma: no cover
    model = model.to(device)
    n_classes = model.get_classes()
    loss_fn = loss()
    model.eval()
    test_losses = []
    test_preds = []
    test_dices = []

    with torch.inference_mode():
        for batch in test_data:
            test_X, test_y = (
                batch["image"].to(device).float(),
                batch["mask"].to(device).float(),
            )
            logits = model(test_X)
            test_pred = F.softmax(logits, dim=n_classes)
            test_loss = loss_fn(logits, test_y)
            test_dice = sum(dice(logits, test_y, n_classes)) / len(test_y)
            test_losses.append(test_loss.item())
            test_preds.append(test_pred.cpu())
            test_dices.append(test_dice)
    return torch.cat(test_preds, dim=0)


def save_model(model, name):
    MODEL_PATH = Path("weights")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = f"{name}.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving {MODEL_NAME} to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
