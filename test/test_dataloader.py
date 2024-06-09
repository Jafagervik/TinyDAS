import unittest

from tinydas.dataloader import DataLoader
from tinydas.dataset import DataSet


class TestDataLoader(unittest.TestCase):
    def test_simple_dataloader(self):
        ds = DataSet()

        dataloader = DataLoader(ds, batch_size=32, shuffle=False)
        self.assertEqual(dataloader.batch_size, 32)
        self.assertEqual(dataloader.data, ds)
        self.assertEqual(dataloader.num_samples, 4)
