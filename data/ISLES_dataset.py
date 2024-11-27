from torch.utils.data import Dataset
import os, glob, torch
import nibabel as nib


class ISLESDataset(Dataset):
    def __init__(
        self, isles_data_dir="data\\ISLES-2022", modality="dwi", sample_data=False
    ):
        # Set images path.

        if sample_data:
            mask_path = os.path.join(
                "data\\isles_22",
                "derivatives",
                "sub-strokecase*",
                "ses-*",
                "sub-strokecase*_ses-*_msk.nii.gz",
            )
            img_path = os.path.join(
                "data\\isles_22",
                "rawdata",
                "sub-strokecase*",
                "ses-*",
                f"sub-strokecase*_ses-*_{modality}.nii.gz",
            )
        else:
            mask_path = os.path.join(
                isles_data_dir,
                "derivatives",
                "sub-strokecase*",
                "ses-*",
                "sub-strokecase*_ses-*_msk.nii.gz",
            )
            img_path = os.path.join(
                isles_data_dir,
                "sub-strokecase*",
                "ses-*",
                "dwi",
                f"sub-strokecase*_ses-*_{modality}.nii.gz",
            )
        self.img_files = glob.glob(img_path)
        self.mask_files = glob.glob(mask_path)
        if not self.check_matching_number_of_files():
            raise Exception(
                f"Number of image files are not matching \n Image files: {len(self.img_files)} \n Mask files: {len(self.mask_files)}"
            )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        try:
            image = nib.load(self.img_files[index]).get_fdata()
            mask_image = nib.load(self.mask_files[index]).get_fdata()
        except Exception as e:
            raise Exception("Trouble loading image at", e)

        return torch.tensor(image), torch.tensor(mask_image)

    def check_matching_number_of_files(self):
        return len(self.img_files) == len(self.mask_files)
