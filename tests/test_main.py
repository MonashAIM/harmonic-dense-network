from unittest import TestCase
from main import get_five, get_four
from models.HarDNet import HarDNet
import torch


class TestMain(TestCase):
    def test_get_five(self):
        self.assertEqual(get_five(), 5)

    def test_get_four(self):
        self.assertEqual(get_four(), 4)

class ModelTest(TestCase):
    def test_model_39DS(self):
        model = HarDNet(arch='39DS')
        self.assertIsNotNone(model)

        x = torch.rand(size=[32, 13, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, torch.Size([32, 1000]))
        self.assertEqual(len(model.layers), 16)
    
    def test_model_68(self):
        model = HarDNet(arch='68')
        self.assertIsNotNone(model)

        x = torch.rand(size=[32, 13, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, torch.Size([32, 1000]))
        self.assertEqual(len(model.layers), 16)

    def test_model_85(self):
        model = HarDNet(arch='85')
        self.assertIsNotNone(model)

        x = torch.rand(size=[32, 13, 112, 112])
        out = model(x)
        self.assertEqual(out.shape, torch.Size([32, 1000]))
        self.assertEqual(len(model.layers), 20)