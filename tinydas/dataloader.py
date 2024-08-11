import random as rnd
from typing import List

from concurrent.futures import ThreadPoolExecutor, as_completed
from tinygrad import Tensor

from tinydas.dataset import Dataset
from tinydas.utils import load_das_file

class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        devices: List[str],
        num_workers: int = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.devices = devices
        self.indices = list(range(len(dataset)))
        self.num_workers = num_workers
        self.shuffle = shuffle
        if self.shuffle: rnd.shuffle(self.indices)
        self.current_index = 0

    def __iter__(self):
        self.current_index = 0
        if self.shuffle: rnd.shuffle(self.indices)
        return self

    def __len__(self) -> int:
        return len(self.indices)

    def __next__(self) -> Tensor:
        if self.current_index >= len(self.indices):
            raise StopIteration

        if self.batch_size == 1:
            curr = self.current_index 
            self.current_index += 1
            # Unsqueeze
            return self._load_single_data(curr).unsqueeze(0).realize()
            
            
            
        else: 
            end_index = min(self.current_index + self.batch_size, len(self.indices))
            batch_indices = self.indices[self.current_index:end_index]

            batch_data = self._load_batch_data(batch_indices)
            self.current_index = end_index

            batch_tensor = Tensor.stack(*batch_data, dim=0)
            return (
                batch_tensor.shard(self.devices, axis=0).realize()
                if len(self.devices) > 1
                else batch_tensor.realize()
            )

    def _load_batch_data(self, batch_indices: List[int]) -> List[Tensor]:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._load_single_data, idx) for idx in batch_indices]

            batch_data = [future.result().realize() for future in as_completed(futures)]
        
        return batch_data

    def _load_single_data(self, idx: int) -> Tensor: return self.dataset[idx]
