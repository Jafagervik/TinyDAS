import os
import numpy as np
import h5py
from torch.utils.data import Dataset
from typing import Optional, List
import torch
import random as rnd

class DASDataset(Dataset):
    def __init__(
        self, 
        train: bool = True, 
        val_split: float = 0.2,
        shuffle: bool = True,
        n: int = 512
    ):
        self.path = "/cluster/home/jorgenaf/TinyDAS/data" if train else  "/cluster/home/jorgenaf/TinyDAS/infer" 
        self.n = n
        self.val_split = val_split
        self.shuffle = shuffle
        self.filenames = self._get_filenames(n)
        self.train_filenames, self.val_filenames = self._split_dataset()

    def __len__(self) -> int: return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        data = self.load_das_file_no_time(filename)
        data = self._apply_normalization(data)
        return torch.from_numpy(data).to(torch.float32)

    def _get_filenames(self, n: Optional[int]) -> List[str]:
        filenames = sorted(os.listdir(self.path))
        filenames = [os.path.join(self.path, f) for f in filenames]
        if n is not None:
            filenames = filenames[:n]
        return filenames
    
    def _split_dataset(self) -> tuple[List[str], List[str]]:
        split_idx = int(len(self.filenames) * (1 - self.val_split))
        return self.filenames[:split_idx], self.filenames[split_idx:]

    @staticmethod
    def load_das_file_no_time(filename: str) -> np.ndarray:
        with h5py.File(filename, "r") as f:
            data = np.array(f["raw"][:]).T
        return data

    @staticmethod
    def _apply_normalization(data: np.ndarray, min_range=0.0, max_range=1.0) -> np.ndarray:
        min_val = np.min(data)
        max_val = np.max(data)
        return min_range + (data - min_val) * (max_range - min_range) / (max_val - min_val)

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

    def __getitem__(self, idx: int):
        return self.parent_dataset[idx]
