from unittest import TestCase

from data.ISLES_dataset import ISLESDataset


class TestISLESDataset(TestCase):
    def test_ISLES_Dataset(self):
        isles_ds = ISLESDataset(sample_data=True)
        assert isles_ds is not None
        assert len(isles_ds) is not None
        assert isles_ds[0] is not None
