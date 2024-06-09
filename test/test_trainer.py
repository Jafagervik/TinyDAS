import unittest

from tinygrad import nn

from tinydas.dataloader import DataLoader
from tinydas.dataset import DataSet
from tinydas.models.ae import AE
from tinydas.trainer import Trainer


class TestTraining(unittest.TestCase):
    def test_simple_training(self):
        devices = ["CLANG"]
        ae = AE()
        ds = DataSet()
        dataloader = DataLoader(ds, batch_size=32, shuffle=False)

        opt = nn.optim.Adam(nn.state.get_parameters(ae), lr=0.001)

        trainer = Trainer(
            ae,
            dataloader,
            devices=devices,
            epochs=10,
            batch_size=32,
            learning_rate=0.001,
            patience=3,
            min_delta=0.0,
            optimizer=opt,
            early_stop=True,
        )

        self.assertEqual(trainer.epochs, 10)
