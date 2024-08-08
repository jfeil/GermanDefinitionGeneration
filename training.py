import os
import subprocess

import click
from rich.progress import Progress


@click.group()
def cli():
    pass


@cli.command()
@click.argument('model_path', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                default='src/model_training/training/experiment_adapter.py')
@click.argument('dataset_path', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                default='src/model_training/datasets/default.py')
@click.option('--checkpoint-dir', type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='output/checkpoint')
@click.option('--model-output-dir', type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='output/model')
@click.option('--adapter-output-dir', type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='output/adapter')
@click.option("--seed", type=int, default=42)
@click.option("--shuffle", type=bool, default=False)
@click.option('--experiment-id', type=int, default=3)
@click.option("--subset-train", type=float, default=-1)
@click.option("--subset-val", type=float, default=-1)
@click.option("--train-batch-size", type=int, default=16)
@click.option("--gradient-accumulation-steps", type=int, default=16)
@click.option("--eval-batch-size", type=int, default=24)
@click.option("--epochs", type=int, default=30)
@click.option("--eval-steps", type=int, default=500)
@click.option("--init-lr", type=float, default=1e-4)
@click.option("--early-stop", type=bool, default=True)
@click.option("--early-stop-steps", type=int, default=3)
@click.option("--weight-decay", type=float, default=0.01)
@click.option("--bf16", type=bool, default=True)
@click.option("--fp16", type=bool, default=False)
@click.option("--keep-checkpoints", type=bool, default=False)
def train(model_path, dataset_path, checkpoint_dir, model_output_dir, adapter_output_dir, seed, shuffle, experiment_id,
          subset_train, subset_val, train_batch_size, gradient_accumulation_steps, eval_batch_size,
          epochs, eval_steps, init_lr, early_stop, early_stop_steps, weight_decay, bf16, fp16, keep_checkpoints):
    """
    Train LLMs

    """
    from adapters import AdapterTrainer
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, EarlyStoppingCallback, Seq2SeqTrainer

    from src.utils import set_seed, import_module_from_path, get_all_class_variables
    from src.ha_utils import HassioCallback
    from src.mlflow_utils import mlflow

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
        mlflow.log_param("1_model_trainable_params", model.num_parameters(only_trainable=True))
        mlflow.log_param("1_model_total_params", model.num_parameters(only_trainable=False))

        callbacks = [HassioCallback]
        if early_stop:
            callbacks.append(EarlyStoppingCallback(early_stop_steps))

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

        training_args = Seq2SeqTrainingArguments(
            learning_rate=init_lr,
            eval_strategy="steps",
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=eval_batch_size,
            logging_steps=10,
            output_dir=checkpoint_dir,
            overwrite_output_dir=True,
            remove_unused_columns=True,
            predict_with_generate=True,
            eval_accumulation_steps=gradient_accumulation_steps,
            eval_steps=eval_steps,
            bf16=bf16,
            fp16=fp16,
            tf32=False,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            seed=seed,
            save_total_limit=5,
            dataloader_num_workers=4,
            # torch_compile=True
        )

        if is_adapter:
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset_train,  # .sort("length", reverse=False),
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
                train_dataset=dataset_train,  # .sort("length", reverse=True),
                eval_dataset=dataset_val,
                # compute_metrics=compute_accuracy,
                data_collator=data_collator,
                tokenizer=tokenizer,
                callbacks=callbacks,
            )

        trainer.train()

        if is_adapter:
            adapter_path = os.path.join(adapter_output_dir, f"adapter_data")
            model.save_adapter(adapter_path, model_module.DefinitionModel.adapter_name)
            mlflow.log_artifact(adapter_path)
        else:
            model_path = os.path.join(model_output_dir, f"model_data")
            model.save_pretrained(model_path)
            mlflow.log_artifact(model_path)
        mlflow.log_input(mlflow.data.from_huggingface(dataset_train, targets='labels'), context="training")
        mlflow.log_input(mlflow.data.from_huggingface(dataset_val, targets='labels'), context="validation")

        if not keep_checkpoints:
            import glob
            import shutil
            for f in glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")):
                shutil.rmtree(f)


def call_external_training(model_path, dataset_path, checkpoint_dir, model_output_dir, adapter_output_dir, seed,
                           shuffle, experiment_id, subset_train, subset_val, train_batch_size,
                           gradient_accumulation_steps, eval_batch_size,
                           epochs, eval_steps, init_lr, early_stop, early_stop_steps, weight_decay, bf16, fp16,
                           keep_checkpoints):
    args = ["training", "train", model_path, dataset_path, '--checkpoint-dir', checkpoint_dir,
            '--model-output-dir', model_output_dir,
            '--adapter-output-dir', adapter_output_dir,
            '--seed', str(seed),
            '--shuffle', str(shuffle),
            '--experiment-id', str(experiment_id),
            '--subset-train', str(subset_train),
            '--subset-val', str(subset_val),
            '--train-batch-size', str(train_batch_size),
            "--gradient-accumulation-steps", str(gradient_accumulation_steps),
            '--eval-batch-size', str(eval_batch_size),
            '--epochs', str(epochs),
            '--eval-steps', str(eval_steps),
            '--init-lr', str(init_lr),
            '--early-stop', str(early_stop),
            '--early-stop-steps', str(early_stop_steps),
            '--weight-decay', str(weight_decay),
            '--bf16', str(bf16),
            '--fp16', str(fp16),
            '--keep-checkpoints', str(keep_checkpoints)
            ]

    subprocess.run(args)


@cli.command()
@click.argument('model_path', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                nargs=-1)
@click.argument('dataset_path', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                nargs=1)
@click.option('--checkpoint-dir', type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='output/checkpoint')
@click.option('--model-output-dir', type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='output/model')
@click.option('--adapter-output-dir', type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='output/adapter')
@click.option("--seed", type=int, default=42)
@click.option("--shuffle", type=bool, default=False)
@click.option('--experiment-id', type=int, default=3)
@click.option("--subset-train", type=float, default=-1)
@click.option("--subset-val", type=float, default=-1)
@click.option("--train-batch-size", type=int, default=12)
@click.option("--gradient-accumulation-steps", type=int, default=16)
@click.option("--eval-batch-size", type=int, default=24)
@click.option("--epochs", type=int, default=30)
@click.option("--eval-steps", type=int, default=500)
@click.option("--init-lr", type=float, default=1e-4)
@click.option("--early-stop", type=bool, default=True)
@click.option("--early-stop-steps", type=int, default=10)
@click.option("--weight-decay", type=float, default=0.01)
@click.option("--bf16", type=bool, default=True)
@click.option("--fp16", type=bool, default=False)
@click.option("--keep-checkpoints", type=bool, default=False)
def train_models(model_path, dataset_path, checkpoint_dir, model_output_dir, adapter_output_dir, seed, shuffle,
                 experiment_id,
                 subset_train, subset_val, train_batch_size, gradient_accumulation_steps, eval_batch_size,
                 epochs, eval_steps, init_lr, early_stop, early_stop_steps, weight_decay, bf16, fp16, keep_checkpoints):
    with Progress() as progress:
        task = progress.add_task("Training", total=len(model_path))
        for model in model_path:
            progress.update(task, description=f"Training model {model}", advance=0)
            call_external_training(model, dataset_path, checkpoint_dir, model_output_dir, adapter_output_dir, seed,
                                   shuffle, experiment_id, subset_train, subset_val, train_batch_size,
                                   gradient_accumulation_steps, eval_batch_size,
                                   epochs, eval_steps, init_lr, early_stop, early_stop_steps, weight_decay, bf16, fp16,
                                   keep_checkpoints)
            progress.update(task, advance=1)


@cli.command()
@click.argument('model_path', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                nargs=1)
@click.argument('dataset_path', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                nargs=-1)
@click.option('--checkpoint-dir', type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='output/checkpoint')
@click.option('--model-output-dir', type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='output/model')
@click.option('--adapter-output-dir', type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='output/adapter')
@click.option("--seed", type=int, default=42)
@click.option("--shuffle", type=bool, default=False)
@click.option('--experiment-id', type=int, default=3)
@click.option("--subset-train", type=float, default=-1)
@click.option("--subset-val", type=float, default=-1)
@click.option("--train-batch-size", type=int, default=12)
@click.option("--gradient-accumulation-steps", type=int, default=16)
@click.option("--eval-batch-size", type=int, default=24)
@click.option("--epochs", type=int, default=30)
@click.option("--eval-steps", type=int, default=500)
@click.option("--init-lr", type=float, default=1e-4)
@click.option("--early-stop", type=bool, default=True)
@click.option("--early-stop-steps", type=int, default=10)
@click.option("--weight-decay", type=float, default=0.01)
@click.option("--bf16", type=bool, default=True)
@click.option("--fp16", type=bool, default=False)
@click.option("--keep-checkpoints", type=bool, default=False)
def train_datasets(model_path, dataset_path, checkpoint_dir, model_output_dir, adapter_output_dir, seed, shuffle,
                   experiment_id,
                   subset_train, subset_val, train_batch_size, gradient_accumulation_steps, eval_batch_size,
                   epochs, eval_steps, init_lr, early_stop, early_stop_steps, weight_decay, bf16, fp16,
                   keep_checkpoints):
    with Progress() as progress:
        task = progress.add_task("Training", total=len(dataset_path))
        for dataset in dataset_path:
            progress.update(task, description=f"Training dataset {dataset}", advance=0)
            call_external_training(model_path, dataset, checkpoint_dir, model_output_dir, adapter_output_dir, seed,
                                   shuffle, experiment_id, subset_train, subset_val, train_batch_size,
                                   gradient_accumulation_steps, eval_batch_size,
                                   epochs, eval_steps, init_lr, early_stop, early_stop_steps, weight_decay, bf16, fp16,
                                   keep_checkpoints)
            progress.update(task, advance=1)


@cli.command()
@click.argument('run_id', type=str, required=True)
@click.argument('model_path', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                nargs=1)
@click.argument('dataset_path', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                nargs=1)
@click.option('--checkpoint-dir', type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='output/checkpoint')
@click.option('--model-output-dir', type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='output/model')
@click.option('--adapter-output-dir', type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='output/adapter')
@click.option("--seed", type=int, default=42)
@click.option("--shuffle", type=bool, default=False)
@click.option("--subset-train", type=float, default=-1)
@click.option("--subset-val", type=float, default=-1)
@click.option("--train-batch-size", type=int, default=12)
@click.option("--eval-batch-size", type=int, default=24)
@click.option("--epochs", type=int, default=30)
@click.option("--eval-steps", type=int, default=500)
@click.option("--init-lr", type=float, default=1e-4)
@click.option("--early-stop", type=bool, default=True)
@click.option("--early-stop-steps", type=int, default=10)
@click.option("--weight-decay", type=float, default=0.01)
@click.option("--bf16", type=bool, default=True)
@click.option("--fp16", type=bool, default=False)
@click.option("--keep-checkpoints", type=bool, default=False)
def resume_training(run_id, model_path, dataset_path, checkpoint_dir, model_output_dir, adapter_output_dir, seed,
                    shuffle, subset_train, subset_val, train_batch_size, eval_batch_size, epochs, eval_steps, init_lr,
                    early_stop, early_stop_steps, weight_decay, bf16, fp16, keep_checkpoints):
    """
    Train LLMs

    """
    from adapters import AdapterTrainer
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, EarlyStoppingCallback, Seq2SeqTrainer

    from src.utils import set_seed, import_module_from_path, get_all_class_variables
    from src.ha_utils import HassioCallback
    from src.mlflow_utils import mlflow

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

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(params)
        mlflow.log_param("1_model_trainable_params", model.num_parameters(only_trainable=True))
        mlflow.log_param("1_model_total_params", model.num_parameters(only_trainable=False))
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
            output_dir=checkpoint_dir,
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
            save_total_limit=5
        )

        if is_adapter:
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset_train.sort("length", reverse=True),
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
                train_dataset=dataset_train.sort("length", reverse=True),
                eval_dataset=dataset_val,
                # compute_metrics=compute_accuracy,
                data_collator=data_collator,
                tokenizer=tokenizer,
                callbacks=callbacks,
            )

        trainer.train(resume_from_checkpoint=True)

        if is_adapter:
            adapter_path = os.path.join(adapter_output_dir, f"adapter_data")
            model.save_adapter(adapter_path, model_module.DefinitionModel.adapter_name)
            mlflow.log_artifact(adapter_path)
        else:
            model_path = os.path.join(model_output_dir, f"model_data")
            model.save_pretrained(model_path)
            mlflow.log_artifact(model_path)

        if not keep_checkpoints:
            import glob
            import shutil
            for f in glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")):
                shutil.rmtree(f)

        mlflow.log_input(mlflow.data.from_huggingface(dataset_train, targets='labels'), context="training")
        mlflow.log_input(mlflow.data.from_huggingface(dataset_val, targets='labels'), context="validation")


if __name__ == '__main__':
    cli()
