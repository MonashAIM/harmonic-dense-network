from monai.data import Dataset, DataLoader
from monai.data.image_reader import PILReader
from monai import transforms
import pytorch_lightning as pl


class CovidDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_properties,
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

    def setup(self, stage=None):
        train_data = []
        val_data = []

        for sample in self.data_properties["training"]:
            if sample["fold"] == self.fold:
                val_data.append({"image": sample["image"], "label": sample["label"]})
            else:
                train_data.append({"image": sample["image"], "label": sample["label"]})

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
            batch_size=5,
            num_workers=self.num_workers,
        )

    def get_train_transform(self):
        train_transform = [
            transforms.LoadImaged(
                reader=PILReader(converter=lambda image: image.convert("L")),
                keys=("image", "label"),
                image_only=True,
                ensure_channel_first=True,
            ),
            transforms.Resized(keys=("image", "label"), spatial_size=(128, 128)),
            # transforms.SqueezeDimd("image", dim=0),
            # transforms.CropForegroundd(["image", "label"], source_key="image"),
            transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
            transforms.AsDiscreted("label", threshold=0.5),
            transforms.ToTensord(["image", "label"], device=self.device),
        ]
        train_transform = transforms.Compose(train_transform)
        return train_transform

    def get_val_transform(self):
        val_transform = [
            transforms.LoadImaged(
                reader=PILReader(converter=lambda image: image.convert("L")),
                keys=("image", "label"),
                image_only=True,
                ensure_channel_first=True,
            ),
            transforms.Resized(keys=("image", "label"), spatial_size=(128, 128)),
            # transforms.SqueezeDimd("image", dim=0),
            transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
            transforms.AsDiscreted("label", threshold=0.5),
            transforms.ToTensord(
                ["image", "label"], allow_missing_keys=True, device=self.device
            ),
        ]
        val_transform = transforms.Compose(val_transform)
        return val_transform