from torch.utils.data import Dataset
import os
import glob
import nibabel as nib


class ISLESDataset(Dataset):
    def __init__(self, isles_data_dir="data"):
        # Set images path.
        dwi_path = os.path.join(
            isles_data_dir,
            "rawdata",
            "sub-strokecase*",
            "ses-0001",
            "sub-strokecase*_ses-0001_dwi.nii.gz",
        )
        adc_path = dwi_path.replace("dwi", "adc")
        flair_path = dwi_path.replace("dwi", "flair")
        mask_path = dwi_path.replace("rawdata", "derivatives").replace("dwi", "msk")
        self.dwi_files = glob.glob(dwi_path)
        self.adc_files = glob.glob(adc_path)
        self.flair_files = glob.glob(flair_path)
        self.mask_files = glob.glob(mask_path)
        if not self.check_matching_number_of_files():
            raise Exception(
                f"Number of image files are not matching \n DWI files: {len(self.dwi_files)} \n ADC files: {len(self.adc_files)} \n FLAIR files: {len(self.flair_files)} \n Mask files: {len(self.mask_files)}"
            )

    def __len__(self):
        return len(self.dwi_files)

    def __getitem__(self, index):
        try:
            dwi_image = nib.load(self.dwi_files[index]).get_fdata()
            adc_image = nib.load(self.adc_files[index]).get_fdata()
            flair_image = nib.load(self.flair_files[index]).get_fdata()
            mask_image = nib.load(self.mask_files[index]).get_fdata()
        except Exception as e:
            raise Exception("Trouble loading imagt at", e)

        return dwi_image, adc_image, flair_image, mask_image

    def check_matching_number_of_files(self):
        return (
            (len(self.dwi_files) == len(self.adc_files))
            and (len(self.adc_files) == len(self.flair_files))
            and (len(self.flair_files) == len(self.mask_files))
        )
