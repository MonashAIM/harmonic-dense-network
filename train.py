from utils.train_utils import hardunet_train_loop
from models.HarDVNet3D import HarDUNet
from data.torchio_dataset import get_isles_22
from torch.nn import functional as F
import dvc.api
import torch
from torch import optim

if __name__ == "__main__":
    params = dvc.api.params_show()

    config_name = params["train"]["config"]
    data = get_isles_22(
        batch_size=1, shuffle=True, sample_data=False, restrict_shape=(1, 112, 112, 73)
    )

    unet = HarDUNet(arch=config_name)
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
        name="hardUNet_local",
    )
