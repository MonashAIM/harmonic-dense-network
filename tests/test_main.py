from unittest import TestCase
from main import get_five, get_four


class TestMain(TestCase):
    def test_get_five(self):
        assert get_five() == 5

    def test_get_four(self):
        assert get_four() == 4
