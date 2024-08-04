import adapters
from adapters import AdapterArguments, setup_adapter_training, MAMConfig
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from typing import Tuple


class DefaultModel:
    model_name = "google/mt5-base"
    tokenizer_legacy = False


class DefaultAdapterModel(DefaultModel):
    adapter_config = "double_seq_bn"
    adapter_name = "definition_base"
    is_adapter = True

    @classmethod
    def create_model(cls) -> Tuple[AutoTokenizer, AutoModel]:
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name, legacy=cls.tokenizer_legacy)
        model = AutoModelForSeq2SeqLM.from_pretrained(cls.model_name)
        adapters.init(model)
        adapter_args = AdapterArguments(train_adapter=True, adapter_config=cls.adapter_config)
        setup_adapter_training(model, adapter_args, cls.adapter_name)

        return tokenizer, model


class DefaultFineTuneModel(DefaultModel):
    is_adapter = False

    @classmethod
    def create_model(cls) -> Tuple[AutoTokenizer, AutoModel]:
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name, legacy=cls.tokenizer_legacy)
        model = AutoModelForSeq2SeqLM.from_pretrained(cls.model_name)

        return tokenizer, model


class DefinitionModel(DefaultAdapterModel):
    """
    Overwrite me :)
    """
    pass
