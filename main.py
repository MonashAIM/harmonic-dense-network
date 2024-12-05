from models.HarDUNet import HarDUNet
from data.torchio_dataset import get_isles_22
from utils.train_utils import hardunet_train_loop
from torch.nn import functional as F
from torch import optim
import torchio as tio
import torch

if __name__ == "__main__":  # pragma: no cover
    data = get_isles_22(
        batch_size=1, shuffle=True, sample_data=False, restrict_shape=(1, 112, 112, 73)
    )

    for batch_idx, batch in enumerate(data):
        print(batch["img"][tio.DATA].shape)
        print(batch["mask"][tio.DATA].shape)
        break

    unet = HarDUNet(arch="39DS")
    lr = 0.1
    loss_fn = F.binary_cross_entropy
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hardunet_train_loop(
        model=unet,
        optim=optim.AdamW(params=unet.parameters(), lr=lr),
        loss_fn=loss_fn,
        device=device,
        train_data=data,
        epochs=1,
    )

