import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import h5py
import numpy as np
from tinygrad.dtype import dtypes
from tinygrad.nn import Tensor

from tinydas.utils import load_das_file

"""
class Dataset:
    def __init__(self, path: str = "./data", transpose: bool = False, max_files: int = -1):
        self.path = path
        self.transpose = transpose
        self.max_files = max_files
        self.filenames = self._get_filtered_filenames()
        self.sample_shape = (625, 2137)

    def __repr__(self) -> str:
        return f"Dataset with {len(self.filenames)} files, sample shape: {self.sample_shape}"

    def __len__(self) -> int:
        return len(self.filenames)

    def _get_filtered_filenames(self) -> List[str]:
        filenames = [entry.path for entry in os.scandir(self.path)]
        filtered_filenames = []

        for filename in filenames:
            if len(filtered_filenames) >= self.max_files > 0:
                break
            try:
                with h5py.File(filename, 'r') as f:
                    data_shape = np.array(f['raw'][:]).shape
                    if data_shape == (2137, 625):
                        filtered_filenames.append(filename)

            except Exception as e:
                print(f"Could not read file {filename}: {e}")

        return filtered_filenames

    def load_file(self, filename: str):
        with h5py.File(filename, 'r') as f:
            data = np.array(f['raw'][:], dtype=np.float32).T
            if self.transpose:
                data = data.T
        return data
"""


class Dataset:
    def __init__(self, path: str = "./data", transpose: bool = False, n: int = -1):
        self.path = path
        self.transpose = transpose
        self.data = self._init_data(n)
        self.rows = self.data["data"].shape[1]
        self.cols = self.data["data"].shape[2]
        self.shape = self.data["data"].shape

    def __repr__(self) -> str:
        return f"{self.data['data'].shape}"

    def __len__(self) -> int:
        return self.data["data"].shape[0]

    def _init_data(self, n: int):
        # if entry.is_file() and entry.name.endswith('.h5')]
        filenames = [entry.path for entry in os.scandir(self.path)]
        if n != -1:
            filenames = filenames[:n]

        with ThreadPoolExecutor() as executor:
            results = list(map(load_das_file, filenames))
        # results = [load_das_file(fs) for fs in filenames]
        results = [tup for tup in results if tup[0].shape == (625, 2137)]

        all_data, all_times = zip(*results)
        all_data_tensor = Tensor(np.stack(all_data), requires_grad=False)
        # all_times_tensor = Tensor(np.stack(all_times), requires_grad=False)

        return {"data": all_data_tensor}  # , "times": all_times_tensor}


if __name__ == "__main__":
    data = Dataset()
    print(data)
    print(data.data["times"].shape)
