from tinygrad.nn import Tensor

import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import h5py

class DataSet():
    def __init__(self, path: str = ".\\data", transpose: bool = False):
        self.path = path
        self.transpose = transpose
        self.data = self._init_data()
        self.rows = self.data["data"].shape[1]
        self.cols = self.data["data"].shape[2]
        self.shape = self.data["data"].shape
        self.raw = self.data["data"]
        self.times = self.data["times"]

    def __repr__(self) -> str:
        return f"{self.data["data"].shape}"

    def __len__(self) -> int :
        return self.data["data"].shape[0]

    def _load_file(self, filename):
        file_path = os.path.join(self.path, filename)
        with h5py.File(file_path, 'r') as f:
            data = np.array(f['raw'][:], dtype=np.float32).T
            times = np.array(f['timestamp'][:])
        if self.transpose: data = data.T
        return data, times
    
    def _init_data(self):
        filenames = [f for f in os.listdir(self.path)]

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._load_file, filenames))

        all_data, all_times = zip(*results)
        all_data_tensor = Tensor(np.stack(all_data))
        all_times_tensor = Tensor(np.stack(all_times))
        
        return {"data": all_data_tensor, "times": all_times_tensor}

