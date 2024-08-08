import logging
import re
from typing import Tuple

from datasets import Dataset

from src.prompting import prompt_pattern
from src.utils import sanitize_context, sanitize_context_word


class DefaultDataset:
    prompt_pattern = f"%s Was ist die Definition von %s? "
    train_path = "/home/jfeil/MasterThesis/dataset/v5_filtered_shuffled/train.parquet"
    val_path = "/home/jfeil/MasterThesis/dataset/v5_filtered_shuffled/val.parquet"
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
        input_text = input_text.replace("&nbsp;", " ")
        return re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', input_text.strip()))

    @classmethod
    def _sanitize_context(cls, input_text: str) -> str:
        return input_text

    @classmethod
    def _sanitize_gt(cls, input_text: str) -> str:
        return input_text

    @staticmethod
    def _sanitize_word(input_text: str) -> str:
        return input_text

    @classmethod
    def _cleanup(cls, examples):
        context_word = [cls._sanitize_word(w) for w in examples['context_word']]
        title = [cls._sanitize_word(w) for w in examples['title']]
        context_sentence = [cls._sanitize_context(w) for w in examples['context_sentence']]
        gt = [cls._sanitize_gt(w) for w in examples['gt']]

        return context_word, title, context_sentence, gt

    @classmethod
    def _preprocessing(cls, examples):
        context_word, title, context_sentence, gt = cls._cleanup(examples)
        input_texts = [prompt_pattern(c, w, pattern=cls.prompt_pattern) for c, w in zip(context_sentence, context_word)]
        inputs = cls.tokenizer(input_texts, max_length=512, truncation=True)
        inputs["labels"] = cls.tokenizer(text_target=gt, max_length=128, truncation=True)["input_ids"]
        inputs["prompt"] = input_texts  # cls.tokenizer.batch_decode(inputs["input_ids"])
        inputs["title"] = title
        inputs["context_word"] = context_word
        inputs["context_sentence"] = context_sentence
        inputs["gt"] = gt  # cls.tokenizer.batch_decode(inputs["labels"])
        inputs["length"] = [len(x)+len(y) for x, y in zip(inputs['input_ids'], inputs["labels"])]

        return inputs

    @classmethod
    def _no_tokenizer_preprocessing(cls, examples):
        context_word, title, context_sentence, gt = cls._cleanup(examples)
        input_texts = [prompt_pattern(c, w, pattern=cls.prompt_pattern) for c, w in zip(context_sentence, context_word)]
        return {"prompt": input_texts, "title": title, "context_word": context_word,
                "context_sentence": context_sentence, "gt": gt}

    @classmethod
    def _prepare_data(cls, dataset: Dataset, cache=True) -> Dataset:
        if cls.tokenizer is None:
            logging.warning("Tokenizer not available, using no tokenizer")
            return dataset.map(cls._no_tokenizer_preprocessing, batched=True, load_from_cache_file=cache)

        return dataset.map(cls._preprocessing, batched=True, load_from_cache_file=cache)


class DefaultTrainValSet(DefaultDataset):
    @classmethod
    def _data_loading(cls, shuffle: bool, seed: int, subset_train: float, subset_val: float) \
            -> Tuple[Dataset, Dataset]:
        # noinspection PyTypeChecker
        dataset_train = Dataset.from_parquet(cls.train_path, split="train")
        # noinspection PyTypeChecker
        dataset_val = Dataset.from_parquet(cls.val_path, split="val")

        if shuffle:
            dataset_train = dataset_train.shuffle(seed=seed).flatten_indices()
            dataset_val = dataset_val.shuffle(seed=seed).flatten_indices()

        if subset_train > 0:
            dataset_train = cls._subset(dataset_train, subset_train)
        if subset_val > 0:
            dataset_val = cls._subset(dataset_val, subset_val)

        return dataset_train, dataset_val

    @classmethod
    def create_dataset(cls, tokenizer, shuffle: bool, seed: int, subset_train: int | float = -1,
                       subset_val: int | float = -1, max_length=250, cache=True) -> Tuple[Dataset, Dataset]:
        cls.tokenizer = tokenizer
        dataset_train, dataset_val = cls._data_loading(shuffle, seed, subset_train, subset_val)
        dataset_train, dataset_val = cls._prepare_data(dataset_train, cache=cache), cls._prepare_data(dataset_val, cache=cache)

        if max_length > 0:
            dataset_train = dataset_train.filter(lambda x: x["length"] <= max_length)
            dataset_val = dataset_val.filter(lambda x: x["length"] <= max_length)
        return dataset_train, dataset_val


class DefaultTestSet(DefaultDataset):
    test_path = "/home/jfeil/MasterThesis/dataset/v5_filtered_shuffled/test.parquet"

    @classmethod
    def _data_loading(cls, shuffle: bool, seed: int, subset_test: float, max_length=250) -> Dataset:
        # noinspection PyTypeChecker
        dataset_test = Dataset.from_parquet(cls.test_path, split="test")

        if shuffle:
            dataset_test = dataset_test.shuffle(seed=seed).flatten_indices()

        if subset_test > 0:
            dataset_test = cls._subset(dataset_test, subset_test)

        if max_length > 0:
            dataset_test = dataset_test.filter(lambda x: x["length"] <= max_length)

        return dataset_test

    @classmethod
    def create_dataset(cls, tokenizer, shuffle: bool, seed: int,
                       subset_test: int | float = -1, cache=True) -> Dataset:
        cls.tokenizer = tokenizer
        dataset_test = cls._data_loading(shuffle, seed, subset_test)

        return cls._prepare_data(dataset_test, cache=cache)


class DefinitionTestSet(DefaultTestSet):
    """
    Overwrite me :)
"""
    pass


class DefinitionDataset(DefaultTrainValSet):
    """
    Overwrite me :)
"""
    pass
