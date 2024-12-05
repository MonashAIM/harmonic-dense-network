import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.functional.segmentation import generalized_dice_score as dice
import torchio as tio
from pathlib import Path

CHANNELS_DIMENSION = 1


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


def prepare_batch(batch, device):
    inputs = batch["img"][tio.DATA].permute(0, 1, 4, 2, 3).to(device).float()
    # foreground = batch['mask'][tio.DATA].permute(0, 1, 4, 2, 3).to(device).float()
    # background = 1 - foreground
    # targets = torch.cat((background, foreground), dim=CHANNELS_DIMENSION).float()
    targets = (
        batch["mask"][tio.DATA].permute(0, 1, 4, 2, 3).to(device).float()
    )  # Not really sure why we need to separate the background and foreground
    return inputs, targets


def hardunet_train_loop(
    model: nn.Module,
    optim: optim,
    loss_fn: F,
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

    for epoch in range(epochs):
        model.train()
        for batch in train_data:
            batch_X, batch_y = prepare_batch(batch, device)
            logits = model(batch_X)
            y_pred = F.softmax(logits, dim=CHANNELS_DIMENSION)
            loss = loss_fn(y_pred, batch_y)
            train_dice = sum(dice(y_pred, batch_y, n_classes)) / len(batch_y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        if scheduler is not None:
            scheduler.step()

        # Validation phase
        if eval_data is not None:
            model.eval()
            val_losses = []
            with torch.inference_mode():
                for batch in eval_data:
                    val_X, val_y = prepare_batch(batch, device)
                    logits = model(batch_X)
                    val_pred = F.softmax(logits, dim=CHANNELS_DIMENSION)
                    val_loss = loss_fn(val_pred, val_y)
                    val_dice = sum(dice(val_pred, val_y, n_classes)) / len(val_y)
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)

        if epoch % checks == 0:
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
    threshold: float = 0.5,
):  # pragma: no cover
    model = model.to(device)
    model.eval()
    test_preds = []
    with torch.inference_mode():
        for test_X, test_y in test_data:
            test_X, test_y = test_X.to(device), test_y.to(device)
            logits = model(test_X)
            test_pred = F.softmax(logits, dim=CHANNELS_DIMENSION)

            if (
                test_pred.shape[1] == 1
            ):  # Assuming a binary segmentation model (single output channel)
                test_pred = torch.sigmoid(test_pred)
                test_pred = (test_pred > threshold).float()
            else:  # For multi-class segmentation (e.g., softmax output)
                test_pred = torch.sigmoid(test_pred)
                test_pred = torch.argmax(
                    F.softmax(test_pred, dim=CHANNELS_DIMENSION), dim=CHANNELS_DIMENSION
                )

            test_preds.append(test_pred.cpu())
    return torch.cat(test_preds, dim=0)


def save_model(model, name):
    MODEL_PATH = Path("weights")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = f"{name}.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving {MODEL_NAME} to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
