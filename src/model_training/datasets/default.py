import re
from typing import Tuple

from datasets import Dataset

from src.prompting import prompt_pattern
from src.utils import sanitize_context, sanitize_context_word


class DefaultDataset:
    prompt_pattern = f"%s Was ist die Definition von %s? "
    tokenizer = None
    extra_special_tokens = []
    extra_tokens = []

    @staticmethod
    def _subset(dataset, subset: float):
        if subset < 1.0:
            subset_value = int(len(dataset) * subset)
        elif subset > 1.0:
            subset_value = int(subset)
        else:
            raise ValueError(f'{subset} has to be larger or smaller than 1.0')

        return dataset.select(range(subset_value))

    @staticmethod
    def _sanitize_spaces(input_text: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', input_text.strip()))

    @classmethod
    def _sanitize_context(cls, input_text: str) -> str:
        input_text = input_text.replace("''", "")
        input_text = cls._sanitize_spaces(input_text)
        input_text = sanitize_context(input_text)
        return input_text

    @staticmethod
    def _sanitize_word(input_text: str) -> str:
        return sanitize_context_word(input_text)

    @classmethod
    def _preprocessing(cls, examples):
        input_texts = [prompt_pattern(cls._sanitize_context(context), cls._sanitize_word(word),
                                      pattern=cls.prompt_pattern) for
                       context, word in zip(examples["context_sentence"], examples["context_word"])]
        inputs = cls.tokenizer(input_texts, max_length=512, truncation=True)
        inputs["labels"] = cls.tokenizer(text_target=[cls._sanitize_context(doc) for doc in examples["gt"]], max_length=128, truncation=True)["input_ids"]
        inputs["debug_text"] = input_texts
        inputs["debug_gt"] = [cls._sanitize_context(doc) for doc in examples["gt"]]

        return inputs

    @classmethod
    def _prepare_data(cls, dataset: Dataset) -> Dataset:
        if cls.tokenizer is None:
            raise ValueError("Tokenizer must be set before preprocessing.")

        return dataset.map(cls._preprocessing, batched=True)


class DefaultTrainValSet(DefaultDataset):
    @classmethod
    def _data_loading(cls, shuffle: bool, seed: int, subset_train: float, subset_val: float) \
            -> Tuple[Dataset, Dataset]:
        # noinspection PyTypeChecker
        dataset_train = Dataset.from_parquet("/home/jfeil/MasterThesis/dataset/v1/train.parquet", split="train")
        # noinspection PyTypeChecker
        dataset_val = Dataset.from_parquet("/home/jfeil/MasterThesis/dataset/v1/val.parquet", split="val")

        if shuffle:
            dataset_train = dataset_train.shuffle(seed=seed)
            dataset_val = dataset_val.shuffle(seed=seed)

        if subset_train > 0:
            dataset_train = cls._subset(dataset_train, subset_train)
        if subset_val > 0:
            dataset_val = cls._subset(dataset_val, subset_val)

        return dataset_train, dataset_val

    @classmethod
    def create_dataset(cls, tokenizer, shuffle: bool, seed: int, subset_train: int | float = -1,
                       subset_val: int | float = -1) -> Tuple[Dataset, Dataset]:
        cls.tokenizer = tokenizer
        dataset_train, dataset_val = cls._data_loading(shuffle, seed, subset_train, subset_val)

        return cls._prepare_data(dataset_train), cls._prepare_data(dataset_val)


class DefaultTestSet(DefaultDataset):
    @classmethod
    def _data_loading(cls, shuffle: bool, seed: int, subset_test: float) -> Dataset:
        # noinspection PyTypeChecker
        dataset_test = Dataset.from_parquet("/home/jfeil/MasterThesis/dataset/v1/test.parquet", split="test")

        if shuffle:
            dataset_test = dataset_test.shuffle(seed=seed)

        if subset_test > 0:
            dataset_test = cls._subset(dataset_test, subset_test)

        return dataset_test

    @classmethod
    def create_dataset(cls, tokenizer, shuffle: bool, seed: int,
                       subset_test: int | float = -1) -> Dataset:
        cls.tokenizer = tokenizer
        dataset_test = cls._data_loading(shuffle, seed, subset_test)

        return cls._prepare_data(dataset_test)


class DefinitionDataset(DefaultTrainValSet):
    """
    Overwrite me :)
"""
    pass
