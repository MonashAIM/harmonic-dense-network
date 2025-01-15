from monai.losses import DiceCELoss
from src.utils.train_utils import HardUnetTrainer
from src.data.ISLES_dataset import ISLESDataModule
from src.data.covid_dataset import CovidDataModule
from src.models.mseg_hardnet import HarDMSEG
import torch
import json
import pytorch_lightning as pl
from dvclive.lightning import DVCLiveLogger
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger


pl.seed_everything(42, workers=True)
# torch.set_default_dtype(torch.float32)

if __name__ == "__main__":  # pragma: no cover
    # with open("./data/dataset.json") as json_file:
    #     data = json.load(json_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.init(project="awa")

    # datamodule = ISLESDataModule(data_properties=data, batch_size=1, device=device)
    with open(fr'.\src\data\covid_dataset.json', 'r') as file:
        data = json.load(file)
    datamodule = CovidDataModule(batch_size=32, device=device, data_properties=data)

    # # Total image to read in. In this case, it's 10 (for both train and val). With split = 0.7, 7 wll go to train and 3 will go to val
    datamodule.setup()

    # #Loadin the data according to the upper parameters
    train_loader = datamodule.train_dataloader()

    for batch_idx, batch in enumerate(train_loader):
        print(batch["image"].shape)
        print(torch.unique(batch["label"]))
        break

    unet = HarDMSEG(arch="68")
    lr = 0.00015
    loss = DiceCELoss

    model = HardUnetTrainer(model=unet, roi_size_h=64, roi_size_w=64, lr=lr)
    logger = WandbLogger()

    # initialize Lightning's trainer.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=400,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=20,
    )

    # train
    trainer.fit(model, datamodule)
    wandb.finish()


    # from monai.inferers import SlidingWindowInferer
    # from monai.metrics import DiceMetric
    # import monai.transforms as transforms
    # roi_size_w = 128
    # roi_size_h = 128
    # inferer = SlidingWindowInferer(roi_size=(roi_size_w, roi_size_h), sw_batch_size=1, overlap=0.5)
    # dice_metric = DiceMetric(reduction="mean_batch", get_not_nans=True)
    # post = transforms.Compose(
    #         [transforms.Activations(sigmoid=True), transforms.AsDiscrete(threshold=0.5)]
    #     )
    # test_data = torch.rand(3,1,128,128)
    # test_label = torch.randint(low=0, high=2, size=(3,1,128,128))
    # out = inferer(test_data, unet)
    # out = post(out)
    # val_dice = dice_metric(out, test_label)
    # mean_dice,_ = dice_metric.aggregate()
    # print(out)
    # print(out.shape)
    # print(val_dice)
    # print(mean_dice)