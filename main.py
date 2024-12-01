from models.HarDNet import HarDNet
from models.HarDUNet import HarDUNet
from data.torchio_dataset import get_isles_22
from utils.train_utils import hardunet_train_loop
from torch import nn
from torch.nn import functional as F
from torch import optim

if __name__ == "__main__":  # pragma: no cover
    data = get_isles_22(batch_size=3, shuffle=True, sample_data=False)
    print(len(data))
    for inputs, targets in data:
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        break

    # model = HarDUNet(arch='39DS')

    # optimizer = optim.AdamW(lr=0.1, params=model.parameters())
    # loss_fn = F.binary_cross_entropy()

    
