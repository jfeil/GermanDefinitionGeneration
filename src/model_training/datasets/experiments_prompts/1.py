from src.model_training.datasets.default import DefaultTrainValSet


class DefinitionDataset(DefaultTrainValSet):
    prompt_pattern = f"\"%s\" Was ist die Definition von %s? "
    pass
