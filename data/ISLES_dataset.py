from monai.data import Dataset, DataLoader
from monai import transforms
import pytorch_lightning as pl
import os


class ISLESDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_properties,
        modality="dwi",
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

        self.train_transform = self.get_train_transform()
        self.val_transform = self.get_val_transform()
        self.train_set = self.val_set = self.test_set = None

    def __len__(self):
        return len(self.img_files)

    def setup(self, train_size=None):
        train_data = []
        val_data = []

        for sample in self.data_properties["training"]:
            if train_size is not None and train_size == len(train_data):
                break

            if sample["fold"] == self.fold:
                altered_data = {
                    "id": sample["id"],
                    "fold": sample["fold"],
                    "image": os.path.join("data", "ISLES-2022", sample["image"][0]),
                    "label": os.path.join("data", "ISLES-2022", sample["label"]),
                }
                val_data.append(altered_data)
            else:
                altered_data = {
                    "id": sample["id"],
                    "fold": sample["fold"],
                    "image": os.path.join("data", "ISLES-2022", sample["image"][0]),
                    "label": os.path.join("data", "ISLES-2022", sample["label"]),
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

    def check_matching_number_of_files(self):
        return len(self.img_files) == len(self.mask_files)

    def get_train_transform(self):
        train_transform = [
            transforms.LoadImaged(
                keys=("image", "label"), image_only=True, ensure_channel_first=True
            ),
            transforms.CropForegroundd(["image", "label"], source_key="image"),
            transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
            transforms.Spacingd(
                ["image", "label"],
                pixdim=(2.0, 2.0, 2.0),
                mode=("bilinear", "bilinear"),
            ),
            transforms.RandSpatialCropd(
                ["image", "label"], roi_size=(64, 64, 64), random_size=False
            ),
            transforms.RandAffined(
                ["image", "label"],
                prob=0.15,
                spatial_size=(64, 64, 64),
                scale_range=[0.3] * 3,
                mode=("bilinear", "bilinear"),
            ),
            # transforms.ResizeWithPadOrCropd(keys=("image", "label"),spatial_size=(64, 64, 64)),
            transforms.AsDiscreted("label", threshold=0.5),
            transforms.ToTensord(["image", "label"], device=self.device),
        ]
        train_transform = transforms.Compose(train_transform)
        return train_transform

    def get_val_transform(self):
        val_transform = [
            transforms.LoadImaged(
                keys=("image", "label"), image_only=True, ensure_channel_first=True
            ),
            transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
            transforms.ToTensord(
                ["image", "label"], allow_missing_keys=True, device=self.device
            ),
        ]
        return transforms.Compose(val_transform)
