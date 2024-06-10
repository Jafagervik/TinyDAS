# TinyDAS - Tinygrad meets PubDAS

## Modules

The following is an explanation of the project structure

### Dataset

Loads the HDF5 data in from the `data` folder and exports it to a pytorch esc dataset

### Dataloader

Generates an iterator for the dataset of a certain batchsize

### Models

```python

class BaseAE(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, X: Tensor) -> Tensor:
        pass

    @abstractmethod
    def criterion(self, X: Tensor) -> Tensor:
        pass

    def predict(self, X: Tensor) -> Tensor:
        return self(X)[0]
```

Most models are based on this abstractclass

### Utils

### Finding anomalies

### Hyperparameters

They are stored in yaml files under the `configs` directory.
Name of the config is the name of the model in lowercase

```yaml
epochs: 10
lr: 0.001
batch_size: 16
```

## How to run

`python main.py -t train -m ae`

or alternatively

`python main.py -t detect -m ae`
