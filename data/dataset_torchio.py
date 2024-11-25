import nibabel as nib
import torchio as tio
import os, glob


def get_isles_22_dwi_subjects(isles_data_dir="data\\ISLES-2022", modality="dwi"):
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
    img_files = glob.glob(img_path)
    mask_files = glob.glob(mask_path)

    assert len(img_files) == len(mask_files)

    subjects = []
    for image_path, label_path in zip(img_files, mask_files):
        subject = tio.Subject(
            dwi=tio.ScalarImage(image_path),
            mask=tio.LabelMap(label_path),
        )
        subjects.append(subject)
    return subjects
