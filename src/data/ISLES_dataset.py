from monai.data import Dataset, DataLoader
from monai import transforms
import pytorch_lightning as pl
import os
import nibabel as nib
import numpy as np
import cv2
from monai.data.image_reader import PILReader
import torch

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

    def setup(self, train_size=None, stage=None):
        train_data = []
        val_data = []
        mask_rejection_threshold = 0.001

        for sample in self.data_properties["training"]:
            if train_size is not None and train_size == len(train_data):
                    break
            image_path = os.path.join(os.getcwd(), "src", "data", "isles_22", sample["image"][0])
            label_path = os.path.join(os.getcwd(), "src", "data", "isles_22", sample["label"])

            image_data = nib.load(image_path).get_fdata()
            label_data = nib.load(label_path).get_fdata()
            x_dim, y_dim, z_dim = image_data.shape

            for i in range(z_dim):
                
                # Extract the 2D slice from the 3D data and mask
                slice_data = image_data[:, :, i]
                mask_slice_data = label_data[:, :, i]

                # Reject if the mask slice has enough nonzero pixels/lesion 
                if np.count_nonzero(mask_slice_data) / mask_slice_data.size >= mask_rejection_threshold:
                    slice_data_cropped = slice_data[10:190, 40:220]
                    slice_data_resized = cv2.resize(slice_data_cropped, (192, 192), interpolation=cv2.INTER_LINEAR)
                    if np.std(slice_data_resized, ddof=1) == 0:
                        break
                    data_norm = (slice_data_resized - np.mean(slice_data_resized)) / np.std(slice_data_resized, ddof=1)
                    data_norm = torch.from_numpy(data_norm)
                    data_norm = torch.unsqueeze(data_norm,0)

                    # Resize the mask
                    mask_slice_data_cropped = mask_slice_data[10:190, 40:220]
                    mask_slice_data_resized = cv2.resize(mask_slice_data_cropped, (192, 192), interpolation=cv2.INTER_NEAREST)
                    mask_slice_data_resized = torch.from_numpy(mask_slice_data_resized)
                    mask_slice_data_resized = torch.unsqueeze(mask_slice_data_resized,0)

                    if sample["fold"] == self.fold:
                        altered_data = {
                            "id": sample["id"],
                            "fold": sample["fold"],
                            "image": data_norm,
                            "label": mask_slice_data_resized,
                        }
                        val_data.append(altered_data)
                    else:
                        altered_data = {
                            "id": sample["id"],
                            "fold": sample["fold"],
                            "image": data_norm,
                            "label": mask_slice_data_resized,
                        }
                        train_data.append(altered_data) 

        self.train_set = Dataset(
            train_data, **self.dataset_kwargs
        )

        self.val_set = Dataset(
            val_data, **self.dataset_kwargs
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
            self.test_set,
            batch_size=1,
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
            transforms.CropForegroundd(["image", "label"], source_key="image"),
            transforms.Spacingd(
                ["image", "label"], pixdim=(1.2, 1.2), mode=("nearest", "nearest")
            ),
            transforms.Resized(keys=("image", "label"), spatial_size=(128, 128)),
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
                reader=PILReader(converter=lambda image: image.convert("L")),
                keys=("image", "label"),
                image_only=True,
                ensure_channel_first=True,
            ),
            transforms.CropForegroundd(["image", "label"], source_key="image"),
            transforms.Spacingd(
                ["image", "label"], pixdim=(1.2, 1.2), mode=("nearest", "nearest")
            ),
            transforms.Resized(keys=("image", "label"), spatial_size=(128, 128)),
            # transforms.SqueezeDimd("image", dim=0),
            # transforms.SqueezeDimd("label", dim=0),
            transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
            transforms.AsDiscreted("label", threshold=0.5),
            transforms.ToTensord(["image", "label"], device=self.device),
            
        ]
        return transforms.Compose(val_transform)
    
if __name__ == '__main__':
    import json
    import torch
    with open(fr'.\src\data\dataset.json', 'r') as file:
        data = json.load(file)
    datamodule = ISLESDataModule(batch_size=32, data_properties=data)

    # # Total image to read in. In this case, it's 10 (for both train and val). With split = 0.7, 7 wll go to train and 3 will go to val
    datamodule.setup()

    # #Loadin the data according to the upper parameters
    train_loader = datamodule.train_dataloader()
    print(len(train_loader))
    for batch_idx, batch in enumerate(train_loader):
        print(batch["image"].shape)
        print(torch.unique(batch["label"]))
        break