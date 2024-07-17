from src.model_training.datasets.default import DefaultTrainValSet


class DefinitionDataset(DefaultTrainValSet):
    train_path = "/home/jfeil/MasterThesis/dataset/v2/train.parquet"
    val_path = "/home/jfeil/MasterThesis/dataset/v2/val.parquet"
    prompt_pattern = f"\"%s\" Was ist die Definition von %s? "
    pass
