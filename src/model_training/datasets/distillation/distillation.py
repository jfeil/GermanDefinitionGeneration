from src.model_training.datasets.experiments_sanitize.complete_sanitization import DefinitionDataset as TemplateTrain
from src.model_training.datasets.experiments_sanitize.complete_sanitization import DefinitionTestSet as TemplateTest


class DefinitionDataset(TemplateTrain):
    train_path = "/home/jfeil/MasterThesis/dataset_distillation/v0/train.parquet"
    val_path = "/home/jfeil/MasterThesis/dataset_distillation/v0/val.parquet"


class DefinitionTestSet(TemplateTest):
    test_path = "/home/jfeil/MasterThesis/dataset_distillation/v0/test.parquet"


if __name__ == '__main__':
    test_set = DefinitionTestSet.create_dataset(None, True, 42)
    pass