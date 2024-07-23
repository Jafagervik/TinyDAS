import unittest

from tinydas.dataset import DataSet


class TestDataset(unittest.TestCase):
    def test_simple_dataset(self):
        dataset = DataSet("../data/")

        self.assertEqual(dataset.path, "../data/")
        self.assertEqual(dataset.transpose, False)
        self.assertEqual(dataset.rows, 100)
        self.assertEqual(dataset.cols, 100)
        self.assertEqual(dataset.shape, (100, 100, 100))
