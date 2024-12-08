from monai.data import Dataset, DataLoader
from monai import transforms
import os
import glob
import torch


class ISLESDataModule:
    def __init__(
        self, modality="dwi", sample_data=False, fold=0, batch_size=2, num_workers=0
    ):
        if sample_data:
            img_path, mask_path = self.get_sample_path(modality=modality)
        else:
            img_path, mask_path = self.get_actual_path(modality=modality)

        self.img_files = glob.glob(img_path)
        self.mask_files = glob.glob(mask_path)
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers

        if not self.check_matching_number_of_files():
            raise Exception(
                f"Number of image files are not matching \n Image files: {len(self.img_files)} \n Mask files: {len(self.mask_files)}"
            )
        self.train_transform = self.get_train_transform()
        self.val_transform = self.get_val_transform()
        self.train_set = self.val_set = self.test_set = None

    def __len__(self):
        return len(self.img_files)

    def setup(self, size=None, split=0.7):
        all_instances = []

        for image_path, label_path in zip(self.img_files, self.mask_files):
            all_instances.append({"image": image_path, "label": label_path})
            if (size is not None) and (len(all_instances) == size):
                break

        train_indices_limiter = int(split * len(all_instances))

        train_data, val_data = (
            all_instances[:train_indices_limiter],
            all_instances[train_indices_limiter:],
        )

        train_set = Dataset(
            train_data,
            transform=self.train_transform,
        )

        val_set = Dataset(
            val_data,
            transform=self.val_transform,
        )

        self.val_loader = DataLoader(
            val_set,
            batch_size=1,
            num_workers=self.num_workers,
        )

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def check_matching_number_of_files(self):
        return len(self.img_files) == len(self.mask_files)

    def get_sample_path(self, modality="dwi"):
        mask_path = os.path.join(
            "data",
            "isles_22",
            "derivatives",
            "sub-strokecase*",
            "ses-*",
            "sub-strokecase*_ses-*_msk.nii.gz",
        )
        img_path = os.path.join(
            "data",
            "isles_22",
            "rawdata",
            "sub-strokecase*",
            "ses-*",
            f"sub-strokecase*_ses-*_{modality}.nii.gz",
        )
        return img_path, mask_path

    def get_actual_path(self, modality="dwi"):
        mask_path = os.path.join(
            "data",
            "ISLES-2022",
            "derivatives",
            "sub-strokecase*",
            "ses-*",
            "sub-strokecase*_ses-*_msk.nii.gz",
        )
        img_path = os.path.join(
            "data",
            "ISLES-2022",
            "sub-strokecase*",
            "ses-*",
            "dwi",
            f"sub-strokecase*_ses-*_{modality}.nii.gz",
        )
        return img_path, mask_path

    def get_train_transform(self):
        train_transform = [
            transforms.LoadImaged(keys=("image", "label"), image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.CropForegroundd(["image", "label"], source_key="image"),
            transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
            transforms.Spacingd(
                ["image", "label"],
                pixdim=(2.0, 2.0, 2.0),
                mode=("bilinear", "bilinear"),
            ),
            # transforms.RandSpatialCropd(
            #     ["image", "label"], roi_size=(64, 64, 64), random_size=False
            # ),
            # transforms.RandAffined(
            #     ["image", "label"],
            #     prob=0.15,
            #     spatial_size=(64, 64, 64),
            #     scale_range=[0.3] * 3,
            #     mode=("bilinear", "bilinear"),
            # ),
            transforms.ResizeWithPadOrCropd(keys=("image", "label"),spatial_size=(64, 64, 64)),
            transforms.AsDiscreted("label", threshold=0.5),
            transforms.ToTensord(["image", "label"]),
        ]
        train_transform = transforms.Compose(train_transform)
        return train_transform

    def get_val_transform(self):
        val_transform = [
            transforms.LoadImaged(keys=("image", "label"), image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
            transforms.ToTensord(["image", "label"], allow_missing_keys=True),
        ]
        return transforms.Compose(val_transform)
