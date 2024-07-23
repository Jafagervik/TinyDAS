import unittest

from tinydas.early_stopping import EarlyStopping


class TestEarlyStopping(unittest.TestCase):
    def test_simple_early_stopping(self):
        early_stopping = EarlyStopping(patience=3, min_delta=0.0)
        losses = [10, 9, 8, 7, 7, 7, 7]
        for i in range(len(losses)):
            early_stopping(losses[i])

        self.assertEqual(early_stopping.patience, 3)
        self.assertEqual(early_stopping.min_delta, 0.0)
        self.assertEqual(early_stopping.best_loss, 7)
        self.assertEqual(early_stopping.counter, 3)
        self.assertEqual(early_stopping.early_stop, True)


if __name__ == "__main__":
    unittest.main()
