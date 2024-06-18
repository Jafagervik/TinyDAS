# TinyDAS - Tinygrad meets PubDAS

## Modules

The following is an explanation of the project structure

### Dataset

Loads the HDF5 data in from the `data` folder and exports it to a pytorch esc dataset

### Dataloader

Generates an iterator for the dataset of a certain batchsize

### Models

```python
class MyAwesomeModel(ABC):
    def __init__(self, kwargs**):
        self.M = kwargs["data"]["M"]
        self.N = kwargs["data"]["N"]

        self.net = [Linear(M*N, 1)]

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        y = x.sequential(self.net)
        return y,

    @abstractmethod
    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        (y, ) = self(x)
        loss = mse(y, x)
        return {"loss": loss}

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

# NOTES:
