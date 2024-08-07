import os
import numpy as np
import h5py
from torch.utils.data import Dataset
from typing import Optional, List
import torch

class DASDataset(Dataset):
    def __init__(self, train: bool = True, n: int = 512):
        self.img_dir = "/cluster/home/jorgenaf/TinyDAS/data" if train else  "/cluster/home/jorgenaf/TinyDAS/infer" 
        self.n = n
        self.filenames = self._get_filenames(n)

    def __len__(self) -> int: return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        data = self.load_das_file_no_time(filename)
        data = self._apply_normalization(data)
        return torch.from_numpy(data).to(torch.float32)

    def _get_filenames(self, n: Optional[int]) -> List[str]:
        filenames = [entry.path for entry in os.scandir(self.img_dir)]
        if n is not None:
            filenames = filenames[:n]
        return filenames

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