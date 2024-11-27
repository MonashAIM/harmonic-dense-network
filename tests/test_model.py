import unittest
from models.HarDNet import HarDNet
from models.HarDUNet import HarDUNet
import torch


class ModelTest(unittest.TestCase):
    def test_model_39DS(self):
        model = HarDNet(arch='39DS')
        self.assertIsNotNone(model)

        x = torch.rand(size=[32, 1, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, torch.Size([32, 1000]))
        #TODO: Determine what happens here
        self.assertEqual(len(model.layers), 15)

    def test_model_68(self):
        model = HarDNet(arch='68')
        self.assertIsNotNone(model)

        x = torch.rand(size=[32, 1, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, torch.Size([32, 1000]))
        #TODO: Determine what happens here
        self.assertEqual(len(model.layers), 17)

    def test_model_85(self):
        model = HarDNet(arch='85')
        self.assertIsNotNone(model)

        x = torch.rand(size=[32, 1, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, torch.Size([32, 1000]))
        self.assertEqual(len(model.layers), 20)

    def test_HarDUNet_39DS(self):
        model = HarDUNet(n_classes=1, arch='39DS')
        self.assertIsNotNone(model)

        x = torch.rand(size=[32, 1, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, x.shape)

    def test_HarDUNet_68(self):
        model = HarDUNet(n_classes=1, arch='68')
        self.assertIsNotNone(model)

        x = torch.rand(size=[32, 1, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, x.shape)

    def test_HarDUNet_85(self):
        model = HarDUNet(n_classes=1, arch='85')
        self.assertIsNotNone(model)

        x = torch.rand(size=[32, 1, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, x.shape)