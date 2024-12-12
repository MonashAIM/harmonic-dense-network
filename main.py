from models.HarDUNet import HarDUNet
from monai.losses import DiceCELoss
from utils.train_utils import hardunet_train_loop, prepare_batch
from torch.nn import functional as F
from torch import optim
import torch
from data.ISLES_dataset import ISLESDataModule
import torch
import json
import pytorch_lightning as pl

pl.seed_everything(42, workers=True)
torch.set_default_dtype(torch.float32)

if __name__ == "__main__":  # pragma: no cover
    with open("./data/dataset.json") as json_file:
        data = json.load(json_file)


    datamodule = ISLESDataModule(data_properties=data, batch_size=1)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hardunet_train_loop(
        model=unet,
        optim=optim.AdamW(params=unet.parameters(), lr=lr),
        loss=loss,
        device=device,
        train_data=train_loader,
        epochs=10,
        checks=2
    )
