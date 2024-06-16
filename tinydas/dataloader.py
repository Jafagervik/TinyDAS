import random as rnd
from typing import List

import numpy as np
from tinygrad import Tensor

from .dataset import Dataset


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        devices: List[str],
        shuffle: bool = False,
    ):
        if shuffle:
            rnd.shuffle(dataset.data["data"])
        self.data = dataset.data["data"]
        # self.times = dataset.data["times"]
        self.batch_size = batch_size
        self.num_samples = dataset.shape[0]
        self.current_index = 0
        self.devices = devices

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration

        end_index = min(self.current_index + self.batch_size, self.num_samples)
        batch_data = self.data[self.current_index : end_index]
        # batch_times = self.times[self.current_index : end_index]
        self.current_index = end_index

        return (
            batch_data.shard(self.devices, axis=0)
            if len(self.devices) > 1
            else batch_data
        )
