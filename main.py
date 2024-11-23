from data.ISLES_dataset import ISLESDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset = ISLESDataset()
    train_ds = DataLoader(dataset, 1, True)
    for dwi_image, adc_image, flair_image, mask_image in train_ds:
        print(dwi_image.shape)
        print(mask_image.shape)


def get_five() -> int:
    return 5


def get_four() -> int:
    return 4
