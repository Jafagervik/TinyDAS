import unittest

from tinygrad import Tensor

from tinydas.selections import Opti, select_optimizer


class TestLosses(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.opti = Opti.ADAM

    def test_selected_optim(self):
        self.assertEqual(self.opti, Opti.ADAM)
