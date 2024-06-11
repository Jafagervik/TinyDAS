import os

import numpy as np
from tinygrad.nn import Tensor

from tinydas.utils import load_das_file

from concurrent.futures import ThreadPoolExecutor


class Dataset:
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

    def _init_data(self, n: int):
        # if entry.is_file() and entry.name.endswith('.h5')]
        filenames = [entry.path for entry in os.scandir(self.path)]
        if n != -1:
            filenames = filenames[:n]

        with ThreadPoolExecutor() as executor:
            results = list(map(load_das_file, filenames))
        #results = [load_das_file(fs) for fs in filenames]
        results = [tup for tup in results if tup[0].shape == (625, 2137)]

        all_data, all_times = zip(*results)
        all_data_tensor = Tensor(np.stack(all_data), requires_grad=True)
        all_times_tensor = Tensor(np.stack(all_times), requires_grad=True)

        return {"data": all_data_tensor, "times": all_times_tensor}


if __name__ == "__main__":
    data = Dataset()
    print(data)
    print(data.data["times"].shape)
