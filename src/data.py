from typing import Tuple
import numpy as np
import h5py

from numpy import ndarray

class DataLoader:
    def __init__(self) -> None:
        pass

    def __call__(self, *args: np.Any, **kwds: np.Any) -> np.Any:
        self._read_das_file()

    def _read_das_file(self, filename: str) -> Tuple[ndarray, ndarray]:
        pass

