from unittest import TestCase
from models.HarDNet import HarDNet
import torch


class ModelTest(TestCase):
    def test_model_39DS(self):
        model = HarDNet(arch='39DS')
        self.assertIsNotNone(model)

        x = torch.rand(size=[32, 13, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, torch.Size([32, 1000]))
        #TODO: Determine what happens here
        #self.assertEqual(len(model.layers), 15)

    def test_model_68(self):
        model = HarDNet(arch='68')
        self.assertIsNotNone(model)

        x = torch.rand(size=[32, 13, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, torch.Size([32, 1000]))
        #TODO: Determine what happens here
        #self.assertEqual(len(model.layers), 15)

    def test_model_85(self):
        model = HarDNet(arch='85')
        self.assertIsNotNone(model)

        x = torch.rand(size=[32, 13, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, torch.Size([32, 1000]))
        self.assertEqual(len(model.layers), 20)