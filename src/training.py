import importlib
import os
import random
import sys

import click
import numpy as np
import torch
from adapters import AdapterTrainer
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, EarlyStoppingCallback, Seq2SeqTrainer

sys.path.insert(0, '../')

from src.ha_utils import HassioCallback
from src.model_training.training.default import DefaultAdapterModel, DefaultFineTuneModel
from src.mlflow_utils import mlflow


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def import_module_from_path(module_path):
    # Ensure the path is absolute
    module_path = os.path.abspath(module_path)

    # Extract module name from path
    module_name = os.path.splitext(os.path.basename(module_path))[0]

    # Check if the module file exists
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"No such file: '{module_path}'")

    # Load the module from the given path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not load spec from '{module_path}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def get_all_class_variables(cls, prefix):
    class_vars = {}
    for base in cls.__mro__:
        for key, value in base.__dict__.items():
            # Filter out methods and special attributes
            if type(value) is not classmethod and not callable(value) and not key.startswith('__'):
                class_vars[prefix + "_" + key] = value
    return class_vars


@click.command()
@click.option("--seed", type=int, default=42)
@click.option("--shuffle", type=bool, default=True)
@click.argument('model_path', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                default='model_training/training/default.py')
@click.argument('dataset_path', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                default='model_training/datasets/default.py')
@click.argument('output_dir', type=click.Path(file_okay=False, dir_okay=True, writable=True),
                default='../training_output/model')
@click.argument('adapter_output', type=click.Path(file_okay=False, dir_okay=True, writable=True),
                default='../training_output/adapter')
@click.option('--experiment-id', type=int, default=3)
@click.option("--subset-train", type=float, default=-1)
@click.option("--subset-val", type=float, default=-1)
@click.option("--train-batch-size", type=int, default=12)
@click.option("--eval-batch-size", type=int, default=12)
@click.option("--epochs", type=int, default=30)
@click.option("--eval-steps", type=int, default=500)
@click.option("--init-lr", type=float, default=1e-4)
@click.option("--early-stop", type=bool, default=True)
@click.option("--early-stop-steps", type=int, default=10)
@click.option("--weight-decay", type=float, default=0.01)
@click.option("--bf16", type=bool, default=True)
@click.option("--fp16", type=int, default=False)
def train(seed, shuffle, model_path, dataset_path, output_dir, adapter_output, experiment_id,
          subset_train, subset_val, train_batch_size, eval_batch_size,
          epochs, eval_steps, init_lr, early_stop, early_stop_steps, weight_decay, bf16, fp16):
    """
    Train LLMs

    """
    set_seed(seed)
    dataset_module = import_module_from_path(dataset_path)
    model_module = import_module_from_path(model_path)
    params = get_all_class_variables(dataset_module.DefinitionDataset, "0_dataset")
    params.update(get_all_class_variables(model_module.DefinitionModel, "1_model"))

    if model_module.DefinitionModel.is_adapter:
        is_adapter = True
    else:
        is_adapter = False

    tokenizer, model = model_module.DefinitionModel.create_model()
    if dataset_module.DefinitionDataset.extra_tokens:
        tokenizer.add_tokens(dataset_module.DefinitionDataset.extra_tokens)
    if dataset_module.DefinitionDataset.extra_special_tokens:
        tokenizer.add_tokens(dataset_module.DefinitionDataset.extra_special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    dataset_train, dataset_val = dataset_module.DefinitionDataset.create_dataset(tokenizer, shuffle=shuffle, seed=seed,
                                                                                 subset_train=subset_train,
                                                                                 subset_val=subset_val)

    print(f"{len(dataset_train)} train samples\n{len(dataset_val)} validation samples")

    print(f"{model.num_parameters(only_trainable=True)} / {model.num_parameters(only_trainable=False)} "
          f"parameters are trainable\n--> "
          f"{model.num_parameters(only_trainable=True) / model.num_parameters(only_trainable=False) * 100:.2f}%")

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(params)
        mlflow.log_input(mlflow.data.from_huggingface(dataset_train, targets='labels'), context="training")
        mlflow.log_input(mlflow.data.from_huggingface(dataset_val, targets='labels'), context="validation")

        callbacks = [HassioCallback]
        if early_stop:
            callbacks.append(EarlyStoppingCallback(early_stop_steps))

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

        training_args = Seq2SeqTrainingArguments(
            learning_rate=init_lr,
            evaluation_strategy="steps",
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            logging_steps=10,
            output_dir=output_dir,
            overwrite_output_dir=True,
            remove_unused_columns=True,
            predict_with_generate=True,
            eval_accumulation_steps=1,
            eval_steps=eval_steps,
            bf16=bf16,
            fp16=fp16,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            seed=seed,
        )

        if is_adapter:
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset_train,
                eval_dataset=dataset_val,
                # compute_metrics=compute_accuracy,
                data_collator=data_collator,
                tokenizer=tokenizer,
                callbacks=callbacks
            )
        else:
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset_train,
                eval_dataset=dataset_val,
                # compute_metrics=compute_accuracy,
                data_collator=data_collator,
                tokenizer=tokenizer,
                callbacks=callbacks
            )

        trainer.train()

        if is_adapter:
            adapter_path = os.path.join(adapter_output, f"{model_module.DefinitionModel.adapter_config}_{mlflow.active_run().info.run_id}_{mlflow.active_run().info.run_name}")
            model.save_adapter(adapter_path, model_module.DefinitionModel.adapter_name)
            mlflow.log_artifact(adapter_path)
        else:
            pass


if __name__ == '__main__':
    train()
