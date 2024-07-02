from src.model_training.datasets.default import DefaultTrainValSet


class DefinitionDataset(DefaultTrainValSet):
    prompt_pattern = f"\"%s\"<DEF>Was ist die Definition von %s?<QUE>"
    extra_special_tokens = ["<DEF>", "<QUE>"]

