import os

from tinydas.utils import load_das_file, zscore, minmax, load_das_file_no_time
from tinydas.enums import Normalization
from tinygrad import Tensor, dtypes
import random as rnd

from typing import Optional, List

class Dataset:
    def __init__(
        self,
        path: str = "./data",
        transpose: bool = False,
        n: Optional[int] = None,
        dtype = dtypes.default_float, # TODO: Change to default type
        normalize: Optional[Normalization] = Normalization.MINMAX,
        val_split: float = 0.2,
        shuffle: bool = True,
    ):
        self.path = path
        self.transpose = transpose
        self.normalize = normalize
        self.dtype = dtypes.default_float
        self.val_split = val_split
        self.shuffle = shuffle
        self.n = n
        self.filenames = self._get_filenames(n)
        self.train_filenames, self.val_filenames = self._split_dataset()

    def __repr__(self) -> str:
        return f"Dataset with {len(self.train_filenames)} training files and {len(self.val_filenames)} validation files"

    def __len__(self) -> int: return len(self.filenames)

    @property
    def shape(self): return (self.n, 625, 2137)

    def __getitem__(self, key) -> Tensor:
        if isinstance(key, int): return self._get_single_item(key)
        if isinstance(key, slice):
            batch_data = [self._get_single_item(i) for i in range(key.start, key.stop)]
            batch_tensor = Tensor.stack(*batch_data, dim=0)
            return batch_tensor.realize()
            #return (
            #    batch_tensor.shard(self.devices, axis=0).realize()
            #    if len(self.devices) > 1
            #    else batch_tensor.realize()
            #)

    def _get_single_item(self, idx: int):
        data = load_das_file_no_time(self.filenames[idx])
        if self.normalize: data = self._apply_normalization(data)
        return data.cast(self.dtype)

    def _get_filenames(self, n: Optional[int]) -> List[str]:
        filenames = sorted(os.listdir(self.path))
        filenames = [os.path.join(self.path, f) for f in filenames]

        if n is not None:
            filenames = filenames[:n]
        return filenames

    def _split_dataset(self) -> tuple[List[str], List[str]]:
        if self.shuffle: rnd.shuffle(self.filenames)
        
        split_idx = int(len(self.filenames) * (1 - self.val_split))
        return self.filenames[:split_idx], self.filenames[split_idx:]

    def _apply_normalization(self, data: Tensor) -> Tensor:
        match (self.normalize):
            case Normalization.MINMAX:
                return minmax(data)
            case Normalization.ZSCORE:
                return zscore(data)
            case _:
                return data

    def get_train_dataset(self):
        return DatasetSplit(self, self.train_filenames)

    def get_val_dataset(self):
        return DatasetSplit(self, self.val_filenames)

class DatasetSplit:
    def __init__(self, parent_dataset: Dataset, filenames: List[str]):
        self.parent_dataset = parent_dataset
        self.filenames = filenames

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tensor:
        filename = self.filenames[idx]
        return self.parent_dataset._apply_normalization(load_das_file_no_time(filename)).cast(self.parent_dataset.dtype)
