# TinyDAS - Tinygrad meets the PubDAS dataset

## How to get data

[See this pdf](https://dev.iris.edu/hq/files/initiatives/das_rcn/DAS-RCN-2022_12_02-Spica.pdf)

Or sign up to GLOBUS and go [here](https://app.globus.org/file-manager?origin_id=706e304c-5def-11ec-9b5c-f9dfb1abb183&origin_path=%2FFORESEE%2F&two_pane=false)
## Modules

The following is an explanation of the project structure

### Dataset

Loads the HDF5 data in from the `data` folder and exports it to a pytorch esc dataset

### Dataloader

Uses parallel workers to load single datafiles in parallel

### Models

See examples in the tinydas/models folder

All autoencoders are based on the BaseAE class

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

* Utils for loss scaling and clipping exist in this repo, but is kinda wonky for training certain models. However, F16 inference is easy:

1. Select model
2. Load model
3. model.half()
