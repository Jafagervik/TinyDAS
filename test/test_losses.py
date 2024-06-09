import unittest

from tinygrad import Tensor

from tinydas.losses import kl_div, mae, mse


class TestLosses(unittest.TestCase):
    def test_kl_div(self):
        return
        # x = Tensor([0.5, 0.5])
        # y = Tensor([0.5, 0.5])
        # self.assertEqual(kl_div(x, y), 0.0)

    def test_mae(self):
        x = Tensor([1, 2, 3])
        y = Tensor([2, 3, 4])
        self.assertEqual(mae(x, y).item(), 1.0)

    def test_mse(self):
        x = Tensor([1, 2, 5])
        y = Tensor([3, 4, 3])
        self.assertEqual(mse(x, y).item(), 4.0)
