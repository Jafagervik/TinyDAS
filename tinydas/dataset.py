import os
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
from tinygrad.nn import Tensor


class DataSet:
    """
    DataSet class to load the data from the .h5 files

    Args:
        - path (str): Path to the data directory
        - transpose (bool): Transpose the data or not
    """

    def __init__(self, path: str = "./data", transpose: bool = False, n: int = -1):
        self.path = path
        self.transpose = transpose
        self.data = self._init_data(n)
        self.rows = self.data["data"].shape[1]
        self.cols = self.data["data"].shape[2]
        self.shape = self.data["data"].shape
        self.raw = self.data["data"]
        self.times = self.data["times"]

    def __repr__(self) -> str:
        return f"{self.data['data'].shape}"

    def __len__(self) -> int:
        return self.data["data"].shape[0]

    def _load_file(self, filename: str):
        with h5py.File(filename, "r") as f:
            data = np.array(f["raw"][:], dtype=np.float32).T
            times = np.array(f["timestamp"][:])
        if self.transpose:
            data = data.T
        return data, times

    def _init_data(self, n: int):
        # if entry.is_file() and entry.name.endswith('.h5')]
        filenames = [entry.path for entry in os.scandir(self.path)]
        if n != -1:
            filenames = filenames[:n]

        # with ThreadPoolExecutor() as executor:
        # results = list(map(self._load_file, filenames))
        results = [self._load_file(fs) for fs in filenames]

        all_data, all_times = zip(*results)
        all_data_tensor = Tensor(np.stack(all_data))
        all_times_tensor = Tensor(np.stack(all_times))

        return {"data": all_data_tensor, "times": all_times_tensor}
