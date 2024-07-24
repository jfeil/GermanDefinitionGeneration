from src.model_training.datasets.default import DefaultTrainValSet as TemplateTrain
from src.model_training.datasets.default import DefaultTestSet as TemplateTest


class DefinitionTestSet(TemplateTest):
    test_path = "/home/jfeil/MasterThesis/dataset/v2/test.parquet"


class DefinitionDataset(TemplateTrain):
    """
    Overwrite me :)
"""
    train_path = "/home/jfeil/MasterThesis/dataset/v2/train.parquet"
    val_path = "/home/jfeil/MasterThesis/dataset/v2/val.parquet"
