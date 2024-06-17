import os
from concurrent.futures import ThreadPoolExecutor

from tinygrad import Tensor, dtypes
import h5py

def load_das_file(filename: str):
    with h5py.File(filename, "r") as f:
        s = Tensor(f["raw"][:]).shape
    return s


class Dataset:
    def __init__(
        self,
        path: str = "./data",
        transpose: bool = False,
        n: int = -1,
        start: int = 16304
    ):
        self.path = path
        self.transpose = transpose
        self.data = self.skips(start=start)

    def skips(self,  start: int):
        filenames = [entry.path for entry in os.scandir(self.path)]
        for f in filenames[start:]:
            s = load_das_file(f)
            if s == (2137, 625): continue
            else: print(f, s)

        print("Done")
        return 0


if __name__ == "__main__":
    ds = Dataset()
    s = print(ds.data)