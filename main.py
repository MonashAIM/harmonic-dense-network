from models.HarDUNet import HarDUNet
from utils.train_utils import hardunet_train_loop, prepare_batch
from torch.nn import functional as F
from torch import optim
import torch
from data.ISLES_dataset import ISLESDataModule
from utils.loss_utils import DiceLoss
import torch.nn as nn
from utils.unet3d import UNet3D

if __name__ == "__main__":  # pragma: no cover
    # data = get_isles_22(
    #     batch_size=1, shuffle=True, sample_data=False, restrict_shape=(1, 112, 112, 73)
    # )

    data = ISLESDataModule(batch_size=1)

    # Total image to read in. In this case, it's 10 (for both train and val). With split = 0.7, 7 wll go to train and 3 will go to val
    data.setup(size=60, split=0.8)

    #Loadin the data according to the upper parameters
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    #Should print out in the format (batch_num, channel = 1, 64,64,64) -> eg (2,1,64,64,64) when batch_size = 2
    print(len(train_loader))
    for batch_idx, batch in enumerate(train_loader):
        print(batch["image"].shape)
        # print(batch["image"][0].shape)
        print(batch["label"].shape)
        # print(batch['image'][0][0][32][48])
        # print(batch['label'][0][0][32][48])
        break

    print(torch.unique(batch["image"]))
    print(torch.unique(batch["label"]))

    unet = HarDUNet(arch="39DS", transformer=False)
    # unet = UNet3D(in_channels=1, num_classes=1)

    lr = 0.01
    loss = DiceLoss
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hardunet_train_loop(
        model=unet,
        optim=optim.AdamW(params=unet.parameters(), lr=lr),
        loss=loss,
        device=device,
        train_data=train_loader,
        epochs=20,
        checks=1
    )
