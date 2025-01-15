from unittest import TestCase

from data.torchio_dataset import get_isles_22


class TestTorchIODataset(TestCase):
    def test_torchio_ISLES_Dataset(self):
        isles_dataloader = get_isles_22(batch_size=1, sample_data=True)
        self.assertIsNotNone(isles_dataloader)
        self.assertIsNotNone(len(isles_dataloader))
