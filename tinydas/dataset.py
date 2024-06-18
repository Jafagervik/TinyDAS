import os

from tinydas.utils import load_das_file, zscore, minmax
from tinydas.enums import Normalization
from tinygrad import Tensor

from typing import Optional, List

class Dataset:
    def __init__(
        self,
        path: str = "./data",
        transpose: bool = False,
        n: Optional[int] = None,
        normalize: Optional[Normalization] = None,
    ):
        self.path = path
        self.transpose = transpose
        self.normalize = normalize
        self.filenames = self._get_filenames(n)

    def __repr__(self) -> str:
        return f"Dataset with {len(self.filenames)} files"

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tensor:
        filename = self.filenames[idx]
        data, _ = load_das_file(filename)
        
        if self.transpose:
            data = data.T
        
        if self.normalize:
            data = self._apply_normalization(data)
        
        return data

    def _get_filenames(self, n: Optional[int]) -> List[str]:
        filenames = [entry.path for entry in os.scandir(self.path)]
        if n is not None:
            filenames = filenames[:n]
        return filenames

    def _apply_normalization(self, data: Tensor) -> Tensor:
        if self.normalize == Normalization.MINMAX:
            return minmax(data)
        elif self.normalize == Normalization.ZSCORE:
            return zscore(data)

"""
class Dataset:
    def __init__(
        self,
        path: str = "./data",
        transpose: bool = False,
        n: Optional[int] = None,
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

    def __getitem__(self, idx: int):
        pass

    def _init_data(self, n: int):
        filenames = [entry.path for entry in os.scandir(self.path)]
        if n is not None:
            filenames = filenames[:n]

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(load_das_file, filenames))

        all_data, _ = zip(*results)
        all_data_tensor = Tensor.stack(*all_data, dim=0)

        return {"data": all_data_tensor}  
"""