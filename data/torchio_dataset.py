import torchio as tio
import os
import glob
from torch.utils.data import DataLoader, random_split


def get_isles_22(
    batch_size=8, shuffle=True, sample_data=False, restrict_shape=(1, 112, 112, 73), size=None, split=None
):
    torch_io_ds = tio.SubjectsDataset(
        get_isles_22_dwi_subjects(
            sample_data=sample_data, restrict_shape=restrict_shape, size=size
        )
    )

    if split is None:
        return DataLoader(torch_io_ds, batch_size=batch_size, shuffle=shuffle)

    train_size = int(split * len(torch_io_ds))
    test_size = len(torch_io_ds) - train_size
    train_dataset, test_dataset = random_split(torch_io_ds, [train_size, test_size])
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_data, test_data


def get_isles_22_dwi_subjects(
    isles_data_dir="data\\ISLES-2022",
    modality="dwi",
    sample_data=False,
    restrict_shape=(1, 112, 112, 73),
    size=None
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

            if (size is not None) and (len(subjects) == size):
                break
    return subjects
