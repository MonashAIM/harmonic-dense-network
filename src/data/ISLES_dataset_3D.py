from monai.data import Dataset, DataLoader
from monai import transforms
import pytorch_lightning as pl
import os
import nibabel as nib
import numpy as np
import cv2
from monai.data.image_reader import PILReader
import torch


class ISLESDataModule_3D(pl.LightningDataModule):
    def __init__(
        self,
        data_properties,
        modalities=["dwi"],
        fold=0,
        batch_size=2,
        num_workers=0,
        device="cpu",
        **kwargs,
    ):
        super().__init__()

        self.device = device
        self.data_properties = data_properties
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = kwargs
        self.modalities = self.get_modalities(modalities)

        self.train_transform = self.get_train_transform()
        self.val_transform = self.get_val_transform()
        self.train_set = self.val_set = self.test_set = None

    def setup(self, train_size=None, stage=None):
        train_data = []
        val_data = []

        for mod in self.modalities:
            for sample in self.data_properties["training"]:
                if train_size is not None and train_size == len(train_data):
                    break

                if sample["fold"] == self.fold:
                    altered_data = {
                        "id": sample["id"],
                        "fold": sample["fold"],
                        "image": os.path.join(
                            os.getcwd(), "src", "data", "isles_22", sample["image"][mod]
                        ),
                        "label": os.path.join(
                            os.getcwd(), "src", "data", "isles_22", sample["label"]
                        ),
                    }
                    val_data.append(altered_data)
                else:
                    altered_data = {
                        "id": sample["id"],
                        "fold": sample["fold"],
                        "image": os.path.join(
                            os.getcwd(), "src", "data", "isles_22", sample["image"][mod]
                        ),
                        "label": os.path.join(
                            os.getcwd(), "src", "data", "isles_22", sample["label"]
                        ),
                    }
                    train_data.append(altered_data)

        self.train_set = Dataset(
            train_data, transform=self.train_transform, **self.dataset_kwargs
        )

        self.val_set = Dataset(
            val_data, transform=self.val_transform, **self.dataset_kwargs
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set_2D,
            batch_size=1,
            num_workers=self.num_workers,
        )

    def get_train_transform(self):
        train_transform = [
            transforms.LoadImaged(
                keys=("image", "label"),
                image_only=True,
                ensure_channel_first=True,
            ),
            transforms.CropForegroundd(["image", "label"], source_key="image"),
            transforms.Spacingd(
                ["image", "label"], pixdim=(1.2, 1.2, 1.2), mode=("nearest", "nearest")
            ),
            transforms.Resized(keys=("image", "label"), spatial_size=(128, 128, 128)),
            # transforms.SqueezeDimd("image", dim=0),
            # transforms.SqueezeDimd("label", dim=0),
            transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
            transforms.AsDiscreted("label", threshold=0.5),
            transforms.ToTensord(["image", "label"], device=self.device),
        ]
        train_transform = transforms.Compose(train_transform)
        return train_transform

    def get_val_transform(self):
        val_transform = [
            transforms.LoadImaged(
                keys=("image", "label"),
                image_only=True,
                ensure_channel_first=True,
            ),
            transforms.CropForegroundd(["image", "label"], source_key="image"),
            transforms.Spacingd(
                ["image", "label"], pixdim=(1.2, 1.2, 1.2), mode=("nearest", "nearest")
            ),
            transforms.Resized(keys=("image", "label"), spatial_size=(128, 128, 128)),
            # transforms.SqueezeDimd("image", dim=0),
            # transforms.SqueezeDimd("label", dim=0),
            transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
            transforms.AsDiscreted("label", threshold=0.5),
            transforms.ToTensord(["image", "label"], device=self.device),
        ]
        return transforms.Compose(val_transform)

    def get_modalities(self, modalities):
        out = []
        if len(modalities) > 3:
            out.append(0)
            return out
        for mod in modalities:
            if mod == "dwi":
                out.append(0)
            elif mod == "adc":
                out.append(1)
            elif mod == "flair":
                out.append(2)
        return out


if __name__ == "__main__":
    import json
    import torch

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    with open(rf".\src\data\ISLES_dataset.json", "r") as file:
        data = json.load(file)
    datamodule = ISLESDataModule_3D(
        batch_size=32, data_properties=data, modalities=["dwi", "adc"]
    )

    # # Total image to read in. In this case, it's 10 (for both train and val). With split = 0.7, 7 wll go to train and 3 will go to val
    datamodule.setup_2D()

    # #Loadin the data according to the upper parameters
    train_loader = datamodule.train_dataloader_2D()
    print(train_loader)
    print(len(train_loader))
    for batch in train_loader:
        print(batch["image"].shape)
        print(torch.unique(batch["label"]))
        break
