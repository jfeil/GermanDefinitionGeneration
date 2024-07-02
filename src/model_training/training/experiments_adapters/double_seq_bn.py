from src.model_training.training.default import DefaultAdapterModel


class DefinitionModel(DefaultAdapterModel):
    adapter_config = "double_seq_bn"
    # "par_bn",
    # "scaled_par_bn",
    # "seq_bn_inv",
    # "double_seq_bn_inv",
    # "compacter",
    # "compacter++",
    # "prefix_tuning",
    # "prefix_tuning_flat",
    # "ia3",
    "mam",
    # "unipelt",
   # "prompt_tuning"
