# Pasta Classification

## Cloning

Make sure to clone with the `--recurse` option

## Populate the dataset

Before using this, you have to preprocess the images found in the pasta_dataset submodule : `python3 preprocess.py`.

After running this, create a dataset folder in this directory and copy the processed the images into it.

## Training the model

The `train.ipynb` file is the notebook version of `main.py`.

It is **heavily** recommended to train the model on a graphics card, using cuda on nvidia and rocm on amd [(docker image)](https://hub.docker.com/r/rocm/tensorflow/)
