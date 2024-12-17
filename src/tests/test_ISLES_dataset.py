from unittest import TestCase

from data.ISLES_dataset import ISLESDataset


class TestISLESDataset(TestCase):
    def test_ISLES_Dataset(self):
        isles_ds = ISLESDataset(sample_data=True)
        self.assertIsNotNone(isles_ds)
        self.assertIsNotNone(len(isles_ds))
        self.assertIsNotNone(isles_ds[0])
