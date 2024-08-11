import os

from tinydas.utils import load_das_file, zscore, minmax, load_das_file_no_time
from tinydas.enums import Normalization
from tinygrad import Tensor, dtypes

from typing import Optional, List

class Dataset:
    def __init__(
        self,
        path: str = "./data",
        transpose: bool = False,
        n: Optional[int] = None,
        dtype = dtypes.float16,
        normalize: Optional[Normalization] = None,
    ):
        self.path = path
        self.transpose = transpose
        self.normalize = normalize
        self.dtype = dtype
        self.filenames = self._get_filenames(n)

    def __repr__(self) -> str:
        return f"Dataset with {len(self.filenames)} files"

    def __len__(self) -> int: return len(self.filenames)

    def __getitem__(self, idx: int) -> Tensor:
        """
        Normalize the data before converting datatype 
        """
        filename = self.filenames[idx]
        data = load_das_file_no_time(filename)
        
        if self.normalize:
            data = self._apply_normalization(data)

        return data.cast(self.dtype)

    def _get_filenames(self, n: Optional[int]) -> List[str]:
        filenames = sorted(os.listdir(self.path))
        filenames = [os.path.join(self.path, f) for f in filenames]

        if n is not None:
            filenames = filenames[:n]
        return filenames

    def _apply_normalization(self, data: Tensor) -> Tensor:
        match (self.normalize):
            case Normalization.MINMAX:
                return minmax(data)
            case Normalization.ZSCORE:
                return zscore(data)
            case _:
                return data 
