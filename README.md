# TinyDAS - Tinygrad meets PubDAS

## Modules

The following is an explanation of the project structure

### Dataset

Loads the HDF5 data in from the `data` folder and exports it to a pytorch esc dataset

### Dataloader

Generates an iterator for the dataset of a certain batchsize

### Models

```python
class MyAwesomeModel(BaseAE):
    def __init__(self, kwargs**):
        self.M = kwargs["data"]["M"]
        self.N = kwargs["data"]["N"]

        self.net = [Linear(M*N, 1), Tensor.tanh]

    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        """In certain models, a forward pass returns multiple values"""
        y = x.sequential(self.net)
        return y,

    @property
    def convolutional(self) -> bool:
        """If set to true, data needs to be reshaped in a different manner"""
        return False

    @abstractmethod
    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        """Stored as a dict in case you want to track several losses"""
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
