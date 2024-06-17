import os
from concurrent.futures import ThreadPoolExecutor

from tinydas.utils import load_das_file, minmax, zscore
from tinydas.enums import Normalization
from tinygrad import Tensor

from typing import Optional


class Dataset:
    def __init__(
        self,
        path: str = "./data",
        transpose: bool = False,
        n: int = -1,
        normalize: Optional[Normalization]= None,
    ):
        self.path = path
        self.transpose = transpose
        self.normalize = normalize
        self.data = self._init_data(n)
        self.rows = self.data["data"].shape[1]
        self.cols = self.data["data"].shape[2]
        self.shape = self.data["data"].shape

    def __repr__(self) -> str:
        return f"{self.data['data'].shape}"

    def __len__(self) -> int:
        return self.data["data"].shape[0]

    def _init_data(self, n: int):
        filenames = [entry.path for entry in os.scandir(self.path)]
        # if entry.is_file() and entry.name.endswith('.h5')]
        if n != -1:
            filenames = filenames[:n]

        #with ThreadPoolExecutor() as executor:
        #    results = list(map(load_das_file, filenames))
        results = [load_das_file(fs) for fs in filenames]
        results = [tup for tup in results if tup[0].shape == (625, 2137)]

        print(len(results))

        all_data, all_times = zip(*results)
        # all_times_tensor = all_times[0].stack(*all_times[1:], dim=0)
        #all_data_tensor = all_data[0].stack(*all_data[1:], dim=0)
        all_data_tensor = Tensor.stack(*all_data, dim=0)

        return {"data": all_data_tensor}  # , "times": all_times_tensor}
