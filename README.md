# TinyDAS - Tinygrad meets the PubDAS dataset

## How to get data

[See this pdf](https://dev.iris.edu/hq/files/initiatives/das_rcn/DAS-RCN-2022_12_02-Spica.pdf)

## Modules

The following is an explanation of the project structure

### Dataset

Loads the HDF5 data in from the `data` folder and exports it to a pytorch esc dataset

### Dataloader

Uses parallel workers to load single datafiles in parallel

### Models

See examples in the tinydas/models folder

All autoencoders are based on the BaseAE class

### Utils

### Finding anomalies

Will upload jupyter notebooks soon

### Hyperparameters

They are stored in yaml files under the `configs` directory.
Name of the config is the name of the model in lowercase

## How to run

`python main.py -t train -m ae`

or alternatively

`python main.py -t detect -m ae`

# NOTES:

* Utils for loss scaling and clipping exist, but is kinda wonky. However, f16 inference is easy:

1. Select model
2. Load model
3. model.half()
