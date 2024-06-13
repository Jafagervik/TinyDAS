import random as rnd
from typing import List

from .dataset import Dataset

from tinygrad import Tensor
import numpy as np

"""
class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)
        self.indices = list(range(self.num_samples))
        if shuffle:
            rnd.shuffle(self.indices)
        self.current_index = 0

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration

        end_index = min(self.current_index + self.batch_size, self.num_samples)
        batch_indices = self.indices[self.current_index:end_index]
        batch_data = [self.dataset.load_file(self.dataset.filenames[idx]) for idx in batch_indices]

        self.current_index = end_index

        # Convert batch_data to a single tensor
        batch_data_tensor = Tensor(np.stack(batch_data), requires_grad=False)

        return batch_data_tensor
"""

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, devices: List[str], shuffle: bool = False):
        if shuffle:
            rnd.shuffle(dataset.data)
        self.data = dataset.data["data"]
        #self.times = dataset.data["times"]
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
        #batch_times = self.times[self.current_index : end_index]
        self.current_index = end_index

        return batch_data.shard(self.devices, axis=0)