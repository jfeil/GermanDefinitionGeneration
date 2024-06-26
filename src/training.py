import importlib
import os
import random
import sys

import click
import numpy as np
import torch


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def import_module_from_path(module_path):
    # Ensure the path is absolute
    module_path = os.path.abspath(module_path)

    # Extract module name from path
    module_name = os.path.splitext(os.path.basename(module_path))[0]

    # Check if the module file exists
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"No such file: '{module_path}'")

    # Load the module from the given path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not load spec from '{module_path}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def get_all_class_variables(cls, prefix):
    class_vars = {}
    for base in cls.__mro__:
        for key, value in base.__dict__.items():
            # Filter out methods and special attributes
            if type(value) is not classmethod and not callable(value) and not key.startswith('__'):
                class_vars[prefix + "_" + key] = value
    return class_vars


@click.command()
@click.option("--seed", type=int, default=42)
@click.option("--shuffle", type=bool, default=True)
@click.argument('model_path', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                default='model_training/training/default.py')
@click.argument('dataset_path', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                default='model_training/datasets/default.py')
@click.argument('experiment', type=int, default=3)
@click.option("--subset-train", type=float, default=-1)
@click.option("--subset-val", type=float, default=-1)
def train(seed, shuffle, model_path, dataset_path, experiment, subset_train, subset_val):
    """
    Train LLMs

    """
    set_seed(seed)
    dataset = import_module_from_path(dataset_path)
    print(get_all_class_variables(dataset.DefinitionDataset, "dataset"))
    return
    dataset_train, dataset_val = dataset.DefinitionDataset.create_dataset(None, shuffle=shuffle, seed=seed,
                                                                          subset_train=subset_train,
                                                                          subset_val=subset_val)


if __name__ == '__main__':
    train()
