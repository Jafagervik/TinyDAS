from tinydas.dataset import DataSet


class DataLoader:
    def __init__(self, dataset: DataSet, batch_size: int):
        self.data = dataset.raw
        self.times = dataset.times
        self.batch_size = batch_size
        self.num_samples = dataset.shape[0]
        self.current_index = 0

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration

        end_index = min(self.current_index + self.batch_size, self.num_samples)
        batch_data = self.data[self.current_index : end_index]
        batch_times = self.times[self.current_index : end_index]
        self.current_index = end_index

        return batch_data, batch_times
