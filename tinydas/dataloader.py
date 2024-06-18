import random as rnd
from typing import List, Optional

from tinygrad import Tensor

from tinydas.dataset import Dataset

class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        devices: List[str],
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.devices = devices
        self.indices = list(range(len(dataset)))
        self.shuffle = shuffle
        if self.shuffle: rnd.shuffle(self.indices)
        self.current_index = 0

    def __iter__(self):
        self.current_index = 0
        if self.shuffle: rnd.shuffle(self.indices)
        return self

    def __next__(self) -> Tensor:
        if self.current_index >= len(self.indices):
            raise StopIteration

        end_index = min(self.current_index + self.batch_size, len(self.indices))
        batch_indices = self.indices[self.current_index:end_index]
        batch_data = [self.dataset[i].realize() for i in batch_indices]
        self.current_index = end_index

        batch_tensor = Tensor.stack(*batch_data, dim=0)
        return (
            batch_tensor.shard(self.devices, axis=0).realize()
            if len(self.devices) > 1
            else batch_tensor.realize()
        )

"""
class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        devices: List[str],
        shuffle: bool = False,
        normalize: Optional[Normalization] = None
    ):
        if shuffle:
            rnd.shuffle(dataset.data["data"])
        self.data = dataset.data["data"]
        self.batch_size = batch_size
        self.num_samples = dataset.shape[0]
        self.current_index = 0
        self.devices = devices
        self.normalize = normalize

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self) -> Tensor:
        if self.current_index >= self.num_samples:
            raise StopIteration

        end_index = min(self.current_index + self.batch_size, self.num_samples)
        batch_data = self.data[self.current_index : end_index]
        self.current_index = end_index

        return (
            batch_data.shard(self.devices, axis=0).realize()
            if len(self.devices) > 1
            else batch_data.realize()
        )

"""