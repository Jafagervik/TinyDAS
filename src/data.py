from typing import Tuple
import numpy as np
import h5py
from tinygrad.nn import Tensor
from tinygrad.helpers import dtypes

from numpy import ndarray

class DataLoader:
    def __init__(self, bs: int, random: bool) -> None:
        self.bs = bs
        self.random = random

    def __call__(self):
        pass

    def _read_das_matrix(self, das_path: str, transpose: bool = False):
        f = h5py.File(das_path, 'r')
        das_data = {
            "data": Tensor(f['raw'][:], dtype=dtypes.float32),
            "times": f['timestamp'][:]
        }

        if transpose:
            das_data["data"] = das_data["data"].transpose()

        f.close()
        return das_data


