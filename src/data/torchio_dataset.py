import torchio as tio
import os
import glob
from torch.utils.data import DataLoader


def get_isles_22(
    batch_size=8, shuffle=True, sample_data=False, restrict_shape=(1, 112, 112, 73)
):
    torch_io_ds = tio.SubjectsDataset(
        get_isles_22_dwi_subjects(
            sample_data=sample_data, restrict_shape=restrict_shape
        )
    )
    return DataLoader(torch_io_ds, batch_size=batch_size, shuffle=shuffle)


def get_isles_22_dwi_subjects(
    isles_data_dir="data\\ISLES-2022",
    modality="dwi",
    sample_data=False,
    restrict_shape=(1, 112, 112, 73),
):
    if sample_data:
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
    else:
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

    img_files = glob.glob(img_path)
    mask_files = glob.glob(mask_path)

    assert len(img_files) == len(mask_files)

    subjects = []
    for image_path, label_path in zip(img_files, mask_files):
        subject = tio.Subject(
            img=tio.ScalarImage(image_path),
            mask=tio.LabelMap(label_path),
        )
        if subject.shape == restrict_shape:
            subjects.append(subject)
    return subjects
