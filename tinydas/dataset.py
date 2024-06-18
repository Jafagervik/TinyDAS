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
        else:
            return data

