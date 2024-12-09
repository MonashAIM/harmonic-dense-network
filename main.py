from models.HarDUNet import HarDUNet
from data.torchio_dataset import get_isles_22
from utils.train_utils import hardunet_train_loop
import torch.nn as nn
from torch import optim
import torchio as tio
import torch
from utils.loss_utils import DiceLoss
from utils.unet import UNet

if __name__ == "__main__":  # pragma: no cover
    train_data, test_data = get_isles_22(
        batch_size=1, shuffle=True, sample_data=False, restrict_shape=(1, 112, 112, 73), size=60, split=0.6
    )
    print(len(train_data))
    for batch_idx, batch in enumerate(train_data):
        print(batch["img"][tio.DATA].shape)
        print(batch["mask"][tio.DATA].shape)
        break

    print(len(test_data))
    for batch_idx, batch in enumerate(test_data):
        print(batch["img"][tio.DATA].shape)
        print(batch["mask"][tio.DATA].shape)
        break

    # unet = HarDUNet(arch="39DS")
    unet = UNet(n_channels=1, n_classes=1)
    lr = 0.1
    loss_fn = DiceLoss
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'

    hardunet_train_loop(
        model=unet,
        optim=optim.AdamW(params=unet.parameters(), lr=lr),
        loss=loss_fn,
        device=device,
        train_data=train_data,
        epochs=20,
        save=True,
        name="test_model",
        checks=1
    )

