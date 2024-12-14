from monai.losses import DiceCELoss
from utils.train_utils import HardUnetTrainer
from data.ISLES_dataset import ISLESDataModule
from data.covid_dataset import CovidDataModule
from models.HarDUNet2D import HarDUNet2D
import torch
import json
import pytorch_lightning as pl

pl.seed_everything(42, workers=True)
# torch.set_default_dtype(torch.float32)

if __name__ == "__main__":  # pragma: no cover
    # with open("./data/dataset.json") as json_file:
    #     data = json.load(json_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # datamodule = ISLESDataModule(data_properties=data, batch_size=1, device=device)
    datamodule = CovidDataModule(batch_size=8, device=device)

    # # Total image to read in. In this case, it's 10 (for both train and val). With split = 0.7, 7 wll go to train and 3 will go to val
    datamodule.setup()

    # #Loadin the data according to the upper parameters
    train_loader = datamodule.train_dataloader()

    for batch_idx, batch in enumerate(train_loader):
        print(batch["image"].shape)
        print(torch.unique(batch["label"]))
        break

    unet = HarDUNet2D(arch="39DS", transformer=False)
    lr = 0.0015
    loss = DiceCELoss

    model = HardUnetTrainer(unet=unet, device=device, model_type=unet.get_model_type())
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="./logs", name="hardunet"
    )  # Just change logs directory

    # initialize Lightning's trainer.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=20,
        logger=tb_logger,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
    )

    # train
    trainer.fit(model, datamodule)
