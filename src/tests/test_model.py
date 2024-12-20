import unittest
from models.HarDNet import HarDNet
from models.HarDVNet3D import HarDUNet3D
from models.HarDUNet2D import HarDUNet2D
import torch


class ModelTest(unittest.TestCase):
    @unittest.skip("Takes too long on pipelie to setup")
    def test_model_39DS(self):
        model = HarDNet(arch="39DS")
        self.assertIsNotNone(model)

        x = torch.rand(size=[3, 1, 73, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, torch.Size([3, 1000]))
        self.assertEqual(len(model.layers), 15)

    @unittest.skip("Takes too long on pipelie to setup")
    def test_model_68(self):
        model = HarDNet(arch="68")
        self.assertIsNotNone(model)

        x = torch.rand(size=[3, 1, 73, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, torch.Size([3, 1000]))
        self.assertEqual(len(model.layers), 17)

    @unittest.skip("Takes too long on pipelie to setup")
    def test_model_85(self):
        model = HarDNet(arch="85")
        self.assertIsNotNone(model)

        x = torch.rand(size=[3, 1, 73, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, torch.Size([3, 1000]))
        self.assertEqual(len(model.layers), 20)

    def test_HarDUNet3D_39DS(self):
        model = HarDUNet3D(n_classes=1, arch="39DS")
        self.assertIsNotNone(model)

        x = torch.rand(size=[3, 1, 73, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, x.shape)

    def test_HarDUNet3D_68(self):
        model = HarDUNet3D(n_classes=1, arch="68")
        self.assertIsNotNone(model)

        x = torch.rand(size=[3, 1, 73, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, x.shape)

    def test_HarDUNet3D_85(self):
        model = HarDUNet3D(n_classes=1, arch="85")
        self.assertIsNotNone(model)

        x = torch.rand(size=[3, 1, 73, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, x.shape)

    def test_HarDUNet2D_39DS(self):
        model = HarDUNet2D(n_classes=1, arch="39DS")
        self.assertIsNotNone(model)

        x = torch.rand(size=[3, 1, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, x.shape)

    def test_HarDUNet2D_68(self):
        model = HarDUNet2D(n_classes=1, arch="68")
        self.assertIsNotNone(model)

        x = torch.rand(size=[3, 1, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, x.shape)

    def test_HarDUNet2D_85(self):
        model = HarDUNet2D(n_classes=1, arch="85")
        self.assertIsNotNone(model)

        x = torch.rand(size=[3, 1, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, x.shape)
