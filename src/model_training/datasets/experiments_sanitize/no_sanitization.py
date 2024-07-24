from src.model_training.datasets.default import DefaultTrainValSet as TemplateTrain


class DefinitionDataset(TemplateTrain):
    train_path = "/home/jfeil/MasterThesis/dataset/v5/train.parquet"
    val_path = "/home/jfeil/MasterThesis/dataset/v5/val.parquet"

    @classmethod
    def _sanitize_context(cls, input_text: str) -> str:
        return input_text

    @classmethod
    def _sanitize_word(cls, input_text: str) -> str:
        return input_text

    @classmethod
    def _sanitize_gt(cls, input_text: str) -> str:
        return input_text

    @classmethod
    def _preprocessing(cls, examples):
        from src.prompting import prompt_pattern
        context_word = [cls._sanitize_word(w) for w in examples['context_word']]
        context_sentence = [cls._sanitize_context(w) for w in examples['context_sentence']]
        gt = [cls._sanitize_gt(w) for w in examples['gt']]
        input_texts = [prompt_pattern(c, w, pattern=cls.prompt_pattern) for c, w in zip(context_sentence, context_word)]
        inputs = cls.tokenizer(input_texts, max_length=512, truncation=True)
        inputs["labels"] = cls.tokenizer(text_target=gt, max_length=128, truncation=True)["input_ids"]
        inputs["prompt"] = input_texts  # cls.tokenizer.batch_decode(inputs["input_ids"])
        inputs["context_word"] = context_word
        inputs["context_sentence"] = context_sentence
        inputs["gt"] = gt  # cls.tokenizer.batch_decode(inputs["labels"])

        return inputs
