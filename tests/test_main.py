from unittest import TestCase
from main import get_five


class TestMain(TestCase):
    def test_get_five(self):
        assert get_five() == 5
