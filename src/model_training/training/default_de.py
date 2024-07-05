from datasets import Dataset

from src.model_training.datasets.default import DefaultTestSet, DefaultTrainValSet


class DefinitionTestSet(DefaultTestSet):
    test_path = "/home/jfeil/MasterThesis/dataset/v2/test.parquet"


class DefinitionDataset(DefaultTrainValSet):
    """
    Overwrite me :)
"""
    train_path = "/home/jfeil/MasterThesis/dataset/v2/train.parquet"
    val_path = "/home/jfeil/MasterThesis/dataset/v2/val.parquet"
