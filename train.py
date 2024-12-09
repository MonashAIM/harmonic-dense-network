from utils.train_utils import hardunet_train_loop
from models import HarDUNet
from data.torchio_dataset import get_isles_22

if __name__ == "__main__":
    isles_loader = get_isles_22()
    hard_UNet_model = HarDUNet()

   # hardunet_train_loop(hard_UNet_model)