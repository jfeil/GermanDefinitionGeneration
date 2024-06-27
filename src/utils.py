import re
from typing import List, Tuple, Dict
import importlib
import os
import random
import sys

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
            name = prefix + "_" + key
            if (name not in class_vars and
                    type(value) is not classmethod and
                    not callable(value) and
                    not key.startswith('__')):
                class_vars[name] = value
    return class_vars


class GT(dict):
    def __init__(self, prompt_data, ground_truth):
        super().__init__(self, title=prompt_data['title'], context_word=sanitize_context(prompt_data['context_word']),
                         context_sentence=prompt_data['context_sentence'], ground_truth=ground_truth)

    def __str__(self):
        return f"{self['context_word']} [{self['title']}] --- {self['context_sentence']} ===> {self['ground_truth']}"

    def __hash__(self):
        return hash((self['title'], self['context_word'], self['context_sentence'], self['ground_truth']))


def del_(inp: str, words: List[str]) -> str:
    for word in words:
        inp = inp.replace(word, "")
    return inp


def strip_(inp: str, words: List[str]) -> str:
    for word in words:
        inp = inp.strip(word)
    return inp


def sanitize_context(context: str) -> str:
    context = re.sub(r'<.*?>', '', context)
    return del_(context, ["?", "!", ":", ";", "‚", "‘", "»", "«", "„", "'",
                          "\"", "”", "“", "›", "‹", "[…]", "…"])


def sanitize_context_word(context) -> str:
    context = del_(context, ["[", "]", "<", ">", "–"])
    return strip_(context, [",", ".", "(", ")", "[", "]", "´", "/", "’", "¡", "¿", "′", "†"])


def sanitize_prediction(pred: str) -> str:
    return del_(pred, ['</s>', '<|eot_id|>'])
