from src.model_training.datasets.default import DefinitionDataset as TemplateTrain
from src.model_training.datasets.default import DefinitionTestSet as TemplateTest


class DefinitionDataset(TemplateTrain):
    train_path = "/home/jfeil/MasterThesis/dataset_distillation/v1/train.parquet"
    val_path = "/home/jfeil/MasterThesis/dataset/v5_filtered_shuffled/val.parquet"


class DefinitionTestSet(TemplateTest):
    test_path = "/home/jfeil/MasterThesis/dataset/v5_filtered_shuffled/test.parquet"


if __name__ == '__main__':
    test_set = DefinitionTestSet.create_dataset(None, True, 42)
    pass