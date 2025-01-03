from src.utils.train_utils import HardUnetTrainer
from src.data.covid_dataset import CovidDataModule
from src.models.HarDUNet2D import HarDUNet2D
from torch.nn import functional as F
import dvc.api
import torch
import json
import pytorch_lightning as pl
from dvclive.lightning import DVCLiveLogger

if __name__ == "__main__":
    params = dvc.api.params_show()

    config_name = params["train"]["config"]
    dataset_name = params["prepare"]["dataset"]
    roi_size_w = params["train"]["roi_size_w"]
    roi_size_h = params["train"]["roi_size_h"]
    batch_size = params["train"]["batch_size"]
    lr = params["train"]["lr"]
    test_batch_size = params["train"]["test_batch_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(f".\src\data\{dataset_name}_dataset.json", "r") as file:
        data = json.load(file)

    # datamodule = ISLESDataModule(data_properties=data, batch_size=1, device=device)
    datamodule = CovidDataModule(
        batch_size=batch_size, device=device, data_properties=data
    )

    # # Total image to read in. In this case, it's 10 (for both train and val). With split = 0.7, 7 wll go to train and 3 will go to val
    datamodule.setup()

    # #Loadin the data according to the upper parameters
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print(
        f"-----------Total number of training batches: {len(train_loader)} ---------------------"
    )
    print(
        f"-----------Total number of validation batches: {len(val_loader)} ---------------------"
    )

    unet = HarDUNet2D(arch=config_name)

    loss_fn = F.binary_cross_entropy
    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = HardUnetTrainer(
        unet=unet, device=device, model_type=unet.get_model_type(), lr=lr
    )
    logger = DVCLiveLogger(log_model=True)

    # initialize Lightning's trainer.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=20,
        logger=logger,
        log_every_n_steps=5,
        check_val_every_n_epoch=5,
    )

    # train
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(f"./src/weights/latest_weight.ckpt")
