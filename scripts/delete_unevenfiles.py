import os
from concurrent.futures import ThreadPoolExecutor

from tinygrad import Tensor, dtypes
import h5py

def load_das_file(filename: str):
    with h5py.File(filename, "r") as f:
        if "timestamp" not in f:
            print(f"Removing {filename} because it does not contain 'timestamp'.")
            os.remove(filename)
        r = len(f["raw"][:])
        c = len(f["raw"][0])
    return r, c

class Dataset:
    def __init__(
        self,
        path: str = "./data",
        transpose: bool = False,
        n: int = 5200,
        start: int = 0
    ):
        self.path = path
        self.transpose = transpose
        self.n = n
        self.data = self.skips(start=start)

    def skips(self, start: int):
        c = 0
        i = 0
        filenames = [entry.path for entry in os.scandir(self.path)]
        for f in filenames[start:]:
            s = load_das_file(f)
            # c += s != (2137, 625)
            # if i % 2000 == 0: print(i + start, c)
            i += 1
            #if i == self.n: return
        print("Done")
        return c

if __name__ == "__main__":
    ds = Dataset()
    s = print(ds.data)
