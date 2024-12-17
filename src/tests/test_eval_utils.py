import unittest
import nibabel as nib
import numpy as np
import os
from unittest import TestCase

from src.utils import eval_utils


class TestEvalUtils(TestCase):
    @unittest.skip("Takes too long on pipelie to setup")
    def test_eval_utils(self):
        isles_data_dir = "data\\isles_22"
        example_case = 9

        # Set images path.
        dwi_path = os.path.join(
            isles_data_dir,
            "rawdata",
            "sub-strokecase{}".format("%04d" % example_case),
            "ses-0001",
            "sub-strokecase{}_ses-0001_dwi.nii.gz".format("%04d" % example_case),
        )
        mask_path = dwi_path.replace("rawdata", "derivatives").replace("dwi", "msk")

        dwi_image = nib.load(dwi_path).get_fdata()
        mask_image = nib.load(mask_path).get_fdata()

        dwi_cutoff = np.percentile(dwi_image[dwi_image > 0], 99)
        segmented_image = dwi_image > dwi_cutoff

        voxel_volume = np.prod(nib.load(dwi_path).header.get_zooms()) / 1000  #

        dice_score = eval_utils.compute_dice(mask_image, segmented_image)

        # Compute absolute volume difference
        voxel_volume = (
            np.prod(nib.load(dwi_path).header.get_zooms()) / 1000
        )  # Get voxel volume
        volume_diff = eval_utils.compute_absolute_volume_difference(
            mask_image, segmented_image, voxel_volume
        )
        abs_ls_diff = eval_utils.compute_absolute_lesion_difference(
            mask_image, segmented_image
        )
        les_f1_count = eval_utils.compute_lesion_f1_score(mask_image, segmented_image)
        self.assertEqual(round(dice_score, 2), 0.17)
        self.assertEqual(volume_diff, 10.32)
        self.assertEqual(abs_ls_diff, 87)
        self.assertEqual(round(les_f1_count, 2), 0.25)
