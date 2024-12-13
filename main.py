from models.HarDUNet3D import HarDUNet
from monai.losses import DiceCELoss
from utils.train_utils import HardUnetTrainer
from data.ISLES_dataset import ISLESDataModule
import torch
import json
import pytorch_lightning as pl

pl.seed_everything(42, workers=True)
# torch.set_default_dtype(torch.float32)

if __name__ == "__main__":  # pragma: no cover
    with open("./data/dataset.json") as json_file:
        data = json.load(json_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datamodule = ISLESDataModule(data_properties=data, batch_size=1, device=device)

    # # Total image to read in. In this case, it's 10 (for both train and val). With split = 0.7, 7 wll go to train and 3 will go to val
    datamodule.setup(train_size=30)

    # #Loadin the data according to the upper parameters
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    for batch_idx, batch in enumerate(train_loader):
        print(batch["image"][0].shape)
        print(torch.unique(batch["label"]))
        break

    unet = HarDUNet(arch="39DS", transformer=False)
    lr = 0.0015
    loss = DiceCELoss

    # unet.cuda()

    # hardunet_train_loop(
    #     model=unet,
    #     optim=optim.AdamW(params=unet.parameters(), lr=lr),
    #     loss=loss,
    #     device=device,
    #     train_data=train_loader,
    #     epochs=10,
    #     checks=2,
    # )
    model = HardUnetTrainer(unet=unet, device=device, model_type=unet.get_model_type())
    tb_logger = pl.loggers.TensorBoardLogger(save_dir="./logs", name="hardunet") # Just change logs directory

    # initialize Lightning's trainer.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=20,
        logger=tb_logger,
        check_val_every_n_epoch=2,
    )

    # train
    trainer.fit(model, datamodule)