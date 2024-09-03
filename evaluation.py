import gc
import itertools
import logging
import os
import subprocess
import tempfile
from collections import defaultdict
from logging import warning

import click
from typing import List, Dict, Callable

import pandas
from datasets import Dataset
from rich.progress import track, Progress
from tqdm.auto import tqdm


def run_metrics(preds, metrics, batch_size, overwrite=False):
    import torch
    df = pandas.DataFrame.from_dict(preds)

    for metric in tqdm(metrics, desc="Evaluating"):
        if metric == "None":
            continue
        metric_results = defaultdict(list)  # type: Dict[str, List[float]]
        cur_metric = standard_metrics[metric]()
        for i in tqdm(range(0, len(df), batch_size), leave=False, desc=metric):
            pred, target = preds['prediction'][i:i + batch_size], preds['gt'][i:i + batch_size]
            result = cur_metric.calc_metric(pred, target)
            for k, v in result.items():
                metric_results[k] += v
        del cur_metric
        torch.cuda.empty_cache()
        gc.collect()
        for k, v in metric_results.items():
            if overwrite and k in df.columns:
                df = df.drop(k, axis=1)
            elif not overwrite and k in df.columns:
                warning(f"Metric {k} is already existing and overwriting is disabled. Ignoring.")
            if k not in df.columns:
                df.insert(len(df.columns), k, v, False)

    return df


def placeholder():
    logging.error("NYI")

    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, List[float]]:
        return {
            'placeholder_precision': [0],
            'placeholder_recall': [0],
            'placeholder_f1': [0]
        }

    return calc_metric


class Bertscore:
    def __init__(self):
        from evaluate import load
        self.metric = load("bertscore")

    def calc_metric(self, pred: List[str], target: List[str]) -> Dict[str, List[float]]:
        scores = self.metric.compute(predictions=pred, references=target, lang="de")
        return {
            'bertscore_precision': scores['precision'],
            'bertscore_recall': scores['recall'],
            'bertscore_f1': scores['f1']
        }


def sacrebleu() -> Callable:
    from evaluate import load
    metric = load("sacrebleu")

    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, List[float]]:
        scores = metric.compute(predictions=pred, references=target)
        return {
            'score': scores['score'],
        }

    return calc_metric


class Bleurt:
    def __init__(self):
        from evaluate import load
        self.metric = load("bleurt", "BLEURT-20")

    def calc_metric(self, pred: List[str], target: List[str]) -> Dict[str, List[float]]:
        scores = self.metric.compute(predictions=pred, references=target)['scores']
        return {
            'bleurt_score': scores
        }


class Meteor:
    def __init__(self):
        from evaluate import load
        self.metric = load("meteor")

    def calc_metric(self, pred: List[str], target: List[str]) -> Dict[str, List[float]]:
        scores = [self.metric.compute(predictions=[p], references=[t])['meteor'] for p, t in zip(pred, target)]
        return {
            'meteor_score': scores
        }


# embedding_model="distilbert-base-german-cased"
class Moverscore:
    def __init__(self, embedding_model="distilbert-base-multilingual-cased"):
        os.environ['MOVERSCORE_MODEL'] = embedding_model

    def calc_metric(self, pred: List[str], target: List[str]) -> Dict[str, List[float]]:
        from moverscore_v2 import word_mover_score
        from collections import defaultdict

        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)

        scores = word_mover_score(target, pred, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1,
                                  remove_subwords=True)
        return {
            "moverscore_score": scores
        }


class Nist:
    def __init__(self):
        from evaluate import load
        self.metric = load("nist_mt")

    def calc_metric(self, pred: List[str], target: List[str]) -> Dict[str, List[float]]:
        scores = self.metric.compute(predictions=pred, references=target)['nist_mt']
        return {
            'nist_score': [scores] * len(target),
        }


class Rouge:
    def __init__(self):
        from evaluate import load
        self.metric = load("rouge")

    def calc_metric(self, pred: List[str], target: List[str]) -> Dict[str, List[float]]:
        scores = self.metric.compute(predictions=pred, references=target, use_aggregator=False)
        return {
            'rouge_1': scores['rouge1'],
            'rouge_2': scores['rouge2'],
            'rouge_L': scores['rougeL'],
            'rouge_Lsum': scores['rougeLsum']
        }


class SentenceEmbeddings:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    # Initialize the evaluator
    def calc_metric(self, pred: List[str], target: List[str]) -> Dict[str, List[float]]:
        from sentence_transformers import util
        embedding_1 = self.model.encode(pred, convert_to_tensor=True)
        embedding_2 = self.model.encode(target, convert_to_tensor=True)
        scores = list(util.pytorch_cos_sim(embedding_1, embedding_2).diagonal().cpu().numpy())
        return {
            "sentence-embedding_score": scores
        }


class Ter:
    def __init__(self):
        from evaluate import load
        self.metric = load("ter")

    def calc_metric(self, pred: List[str], target: List[str]) -> Dict[str, List[float]]:
        scores = self.metric.compute(predictions=pred, references=target)
        return {
            'ter_score': scores['score'],
        }


standard_metrics = {
    'BERTScore': Bertscore,
    # 'SacreBLEU': Sacrebleu,
    'BLEURT': Bleurt,
    'METEOR': Meteor,
    'Moverscore': Moverscore,
    # 'NIST': Nist,
    'ROUGE': Rouge,
    'SentenceEmbeddings': SentenceEmbeddings,
    # 'TER': Ter
    'None': None,
}


@click.group()
def cli():
    pass


def load_model(run, device):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from src.mlflow_utils import mlflow, download_run_artifact

    run_data = mlflow.get_run(run)
    if "1_model_is_adapter" not in run_data.data.params:
        raise ValueError("Missing adapter information")
    if "1_model_model_name" not in run_data.data.params:
        raise ValueError("Missing model name information")
    if "0_dataset_prompt_pattern" not in run_data.data.params:
        raise ValueError("Missing dataset prompt pattern information")
    if "1_model_tokenizer_legacy" not in run_data.data.params:
        raise ValueError("Missing tokenizer legacy information")

    model_name = run_data.data.params["1_model_model_name"]
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=run_data.data.params["1_model_tokenizer_legacy"])

    if run_data.data.params['0_dataset_extra_tokens']:
        tokenizer.add_tokens(eval(run_data.data.params['0_dataset_extra_tokens']))
    if run_data.data.params['0_dataset_extra_special_tokens']:
        tokenizer.add_tokens(eval(run_data.data.params['0_dataset_extra_special_tokens']))
    model.resize_token_embeddings(len(tokenizer))

    if run_data.data.params["1_model_is_adapter"] == "True":
        import adapters

        if "1_model_adapter_name" not in run_data.data.params:
            raise ValueError("Missing model name information")
        adapter_name = run_data.data.params["1_model_adapter_name"]
        adapters.init(model)

        # adapter needs to be reinstated
        with tempfile.TemporaryDirectory() as temp_path:
            adapter_path = download_run_artifact(run, "adapter_data", temp_path)
            model.load_adapter(adapter_path)
        model.set_active_adapters(adapter_name)
        logging.info("Adapter '%s' loaded." % adapter_name)
    elif run_data.data.params["1_model_is_adapter"] == "False":
        # model needs to be reinstated
        with tempfile.TemporaryDirectory() as temp_path:
            weight_path = download_run_artifact(run, "model_data", temp_path)
            model = model.from_pretrained(weight_path)
        logging.info("Fine-tuned Model loaded.")
    else:
        raise ValueError("Invalid adapter information")

    model.eval()
    model.to(device)

    return model, tokenizer, run_data.data.params["0_dataset_prompt_pattern"]


@cli.command()
@click.argument('run', type=str)
@click.argument('test_set', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                default='src/model_training/datasets/default_de.py')
@click.option('-m',
              '--metrics',
              type=click.Choice(list(standard_metrics.keys()), case_sensitive=False),
              default=standard_metrics.keys(),
              multiple=True)
@click.option('--batch-size', type=int, default=12)
@click.option("--seed", type=int, default=42)
@click.option("--shuffle", type=bool, default=True)
@click.option("--subset-test", type=float, default=-1)
@click.option("--use-train", type=bool, default=False)
@click.option("--output", type=click.Path(exists=False, file_okay=True, dir_okay=False), default=None)
@click.option('--debug', type=bool, default=False)
def evaluate(run, test_set, metrics, batch_size, seed, shuffle, subset_test, use_train, output, debug):
    import torch
    from transformers import pipeline
    from src.utils import import_module_from_path
    from src.mlflow_utils import mlflow

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    model, tokenizer, prompt_pattern = load_model(run, device)
    if use_train:
        dataset_module = import_module_from_path(test_set)

        dataset_module.DefinitionDataset.prompt_pattern = prompt_pattern
        logging.info(f"Dataset configuration {test_set}")
        logging.info(f"Loaded from {dataset_module.DefinitionDataset.train_path}")

        dataset_test, _ = dataset_module.DefinitionDataset.create_dataset(tokenizer, shuffle=shuffle, seed=seed,
                                                                          subset_train=subset_test)
    else:
        dataset_module = import_module_from_path(test_set)

        dataset_module.DefinitionTestSet.prompt_pattern = prompt_pattern
        logging.info(f"Dataset configuration {test_set}")
        logging.info(f"Loaded from {dataset_module.DefinitionTestSet.test_path}")

        dataset_test = dataset_module.DefinitionTestSet.create_dataset(tokenizer, shuffle=shuffle, seed=seed,
                                                                       subset_test=subset_test)

    logging.info(f"{len(dataset_test)} test examples loaded.")
    pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=device)

    test_predictions = {
        'title': dataset_test['title'],
        'context_sentence': dataset_test['context_sentence'],
        'context_word': dataset_test['context_word'],
        'gt': dataset_test['gt'],
        'prediction': []
    }

    def data_generator():
        for item in dataset_test:
            yield item["prompt"]

    from tqdm.auto import tqdm
    for out in tqdm(pipe(data_generator(), batch_size=batch_size*3, max_length=50, num_beams=5, early_stopping=True),
                    total=len(dataset_test), desc="Inferencing"):
        assert len(out) == 1
        test_predictions['prediction'].append(out[0]["generated_text"])

    del pipe, model, tokenizer
    torch.cuda.empty_cache()

    df = run_metrics(test_predictions, metrics, batch_size)

    if output:
        df.to_json(output, orient="records", lines=True)
    else:
        with mlflow.start_run(run_id=run):
            mlflow.log_table(df, f"evaluation_{dataset_test._fingerprint}.json")
            logging.info(f"{run} data logged!")


@cli.command()
@click.argument('test_set', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                default='/home/jfeil/MasterThesis/dataset_distillation/v0/test.parquet')
@click.option('-m',
              '--metrics',
              type=click.Choice(list(standard_metrics.keys()), case_sensitive=False),
              default=standard_metrics.keys(),
              multiple=True)
@click.option('--batch-size', type=int, default=32)
@click.option("--seed", type=int, default=42)
@click.option("--shuffle", type=bool, default=True)
@click.option("--output", type=click.Path(exists=False, file_okay=True, dir_okay=False), default=None)
@click.option('--debug', type=bool, default=False)
@click.option("--experiment", type=int, default=0)
@click.option("--dataset-fingerprint", type=str, default=None)
def evaluate_parquet(test_set, metrics, batch_size, seed, shuffle, output, debug, experiment, dataset_fingerprint):
    import torch
    from src.mlflow_utils import mlflow

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    dataset_test = Dataset.from_parquet(test_set)
    if shuffle:
        dataset_test = dataset_test.shuffle(seed=seed)

    test_predictions = {
        'title': dataset_test['title'],
        'context_sentence': dataset_test['context_sentence'],
        'context_word': dataset_test['context_word'],
        'gt': dataset_test['wiktionary_gt'],
        'prediction': dataset_test['gt']
    }

    fingerprint = dataset_test._fingerprint if dataset_fingerprint is None else dataset_fingerprint

    df = run_metrics(test_predictions, metrics, batch_size)
    if output:
        df.to_json(output, orient="records", lines=True)
    else:
        with mlflow.start_run(experiment_id=experiment):
            mlflow.log_table(df, f"evaluation_{fingerprint}.json")
            logging.info(f"data logged!")


@cli.command()
@click.argument('test_set', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                default='src/model_training/datasets/default_de.py')
@click.option('-e', '--experiment', 'experiments', type=int, multiple=True, default=[])
@click.option('-r', '--run', 'selected_runs', type=str, multiple=True, default=[])
@click.option('--exclude', 'excluded_runs', type=str, multiple=True, default=[])
@click.option('-m',
              '--metrics',
              type=click.Choice(list(standard_metrics.keys()), case_sensitive=False),
              default=standard_metrics.keys(),
              multiple=True)
@click.option('--batch-size', type=int, default=12)
@click.option("--seed", type=int, default=42)
@click.option("--shuffle", type=bool, default=True)
@click.option("--subset-test", type=float, default=-1)
@click.option('--debug', type=bool, default=False)
def batch_evaluate(test_set, experiments, selected_runs, excluded_runs, metrics, batch_size, seed, shuffle, subset_test, debug):
    from src.mlflow_utils import get_run_list

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    selected_runs += tuple(get_run_list(experiments))
    selected_runs = list(set(selected_runs))

    for run in excluded_runs:
        if run in selected_runs:
            selected_runs.remove(run)

    logging.info("selected runs: %s", selected_runs)
    logging.info("selected metrics: %s", metrics)

    metrics_args = list(sum([("-m", m) for m in metrics], ()))
    with Progress() as progress:
        task = progress.add_task("Evaluating", total=len(selected_runs))
        for run in selected_runs:
            progress.update(task, description=f"Evaluating {run}", advance=0)
            subprocess.run(
                ["python3", "evaluation.py", 'evaluate', str(run), str(test_set), *metrics_args, '--batch-size',
                 str(batch_size), '--seed', str(seed), '--shuffle', str(shuffle), '--subset-test',
                 str(subset_test), '--debug', str(debug)])
            progress.update(task, advance=1)


def debug_metrics():
    pred = ["tiefes, unbedingtes und unerschütterliches Gefühl",
            "Eine starke, tiefe und unbedingte Zuneigung oder Sympathie, die zwischen Menschen, Tieren oder Dingen "
            "bestehen kann und oft durch Gefühle wie Zärtlichkeit, Treue, Hingabe und Akzeptanz gekennzeichnet ist.",
            "Junger Soldat",
            "kurze, dünne, starre Holz- oder Kunststoffstücke",
            "Fertigungen, die als Griff oder als Verbindungselement zwischen verschiedenen Teilen eines Geräts oder "
            "einer Maschine dienen."]
    target = ["inniges Gefühl der Zuneigung für jemanden oder für etwas",
              "inniges Gefühl der Zuneigung für jemanden oder für etwas",
              "Rekrut",
              "Rekrut",
              "Rekrut"]

    for name, metric in standard_metrics.items():
        result = metric().calc_metric(pred, target)
        print(f"{name}: {result}")
        for key, val in result.items():
            assert len(val) == len(target), f"{key} for {name} are not enough outputs"


@cli.command()
@click.argument('run', type=str)
@click.argument('test_set', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                default='src/model_training/datasets/default.py')
@click.option('--experiment', type=int)
@click.option('--batch-size', type=int, default=32)
@click.option("--seed", type=int, default=42)
@click.option("--shuffle", type=bool, default=True)
@click.option("--subset-test", type=float, default=-1)
@click.option("--output", type=click.Path(exists=False, file_okay=True, dir_okay=False), default=None)
@click.option('--debug', type=bool, default=False)
def parameter_tuning(run, test_set, experiment, batch_size, seed, shuffle, subset_test, output, debug):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from src.utils import import_module_from_path
    from src.mlflow_utils import mlflow, download_run_artifact
    from tqdm.auto import tqdm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    model, tokenizer, prompt_pattern = load_model(run, device)
    dataset_module = import_module_from_path(test_set)

    dataset_module.DefinitionTestSet.prompt_pattern = prompt_pattern
    logging.info(f"Dataset configuration {test_set}")
    logging.info(f"Loaded from {dataset_module.DefinitionTestSet.test_path}")

    dataset_test = dataset_module.DefinitionTestSet.create_dataset(tokenizer, shuffle=shuffle, seed=seed,
                                                                   subset_test=subset_test)
    logging.info(f"{len(dataset_test)} test examples loaded.")
    pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=device)

    strategies = [
        {"name": "Greedy search"},
        {"name": "Contrastive search", "penalty_alpha": 0.6, "top_k": 4, "max_new_tokens": 100},
        {"name": "Multinomial search", "do_sample": True, "num_beams": 1, "max_new_tokens": 100},
        {"name": "Beam-search decoding", "num_beams": 5, "max_new_tokens": 50},
        {"name": "Beam-search multinomial sampling", "do_sample": True, "num_beams": 5},
        {"name": "Diverse beam search decoding", "num_beams": 5, "num_beam_groups": 5, "max_new_tokens": 30,
         "diversity_penalty": 1.0},
        {"name": "Repetition penalty", "num_beams": 5, "max_new_tokens": 50, "repetition_penalty": 1.1},
        {"name": "Repetition penalty", "num_beams": 5, "max_new_tokens": 50, "repetition_penalty": 1.2},
        {"name": "Repetition penalty", "num_beams": 5, "max_new_tokens": 50, "repetition_penalty": 1.3},
        {"name": "Repetition penalty", "num_beams": 5, "max_new_tokens": 50, "repetition_penalty": 1.4},
        {"name": "Repetition penalty", "num_beams": 5, "max_new_tokens": 50, "repetition_penalty": 1.5},
        {"name": "Encoder Repetition penalty", "num_beams": 5, "max_new_tokens": 50, "encoder_repetition_penalty": 1.1},
        {"name": "Encoder Repetition penalty", "num_beams": 5, "max_new_tokens": 50, "encoder_repetition_penalty": 1.2},
        {"name": "Encoder Repetition penalty", "num_beams": 5, "max_new_tokens": 50, "encoder_repetition_penalty": 1.3},
        {"name": "Encoder Repetition penalty", "num_beams": 5, "max_new_tokens": 50, "encoder_repetition_penalty": 1.4},
        {"name": "Encoder Repetition penalty", "num_beams": 5, "max_new_tokens": 50, "encoder_repetition_penalty": 1.5},
        {"name": "Temperature", "do_sample": True, "num_beams": 5, "max_new_tokens": 50, "temperature": 0.01},
        {"name": "Temperature", "do_sample": True, "num_beams": 5, "max_new_tokens": 50, "temperature": 0.1},
        {"name": "Temperature", "do_sample": True, "num_beams": 5, "max_new_tokens": 50, "temperature": 0.3},
        {"name": "Temperature", "do_sample": True, "num_beams": 5, "max_new_tokens": 50, "temperature": 0.5},
        {"name": "Temperature", "do_sample": True, "num_beams": 5, "max_new_tokens": 50, "temperature": 0.7},
        {"name": "Temperature", "do_sample": True, "num_beams": 5, "max_new_tokens": 50, "temperature": 0.9},
        {"name": "Temperature", "do_sample": True, "num_beams": 5, "max_new_tokens": 50, "temperature": 1.1},
        {"name": "Temperature", "do_sample": True, "num_beams": 5, "max_new_tokens": 50, "temperature": 1.2},
        {"name": "Temperature", "do_sample": True, "num_beams": 5, "max_new_tokens": 50, "temperature": 1.4},
        {"name": "Temperature", "do_sample": True, "num_beams": 5, "max_new_tokens": 50, "temperature": 1.5},
    ]

    def data_generator():
        for item in dataset_test:
            yield item["prompt"]

    for strategy in tqdm(strategies):
        test_predictions = {
            'title': dataset_test['title'],
            'context_sentence': dataset_test['context_sentence'],
            'context_word': dataset_test['context_word'],
            'gt': dataset_test['gt'],
            'prediction': []
        }

        logging.info("Currently used strategy: %s" % strategy)
        used_strategy = dict(strategy)
        name = used_strategy.pop("name")
        for out in tqdm(pipe(data_generator(), batch_size=batch_size, **used_strategy),
                        total=len(dataset_test), desc="Inferencing"):
            assert len(out) == 1
            test_predictions['prediction'].append(out[0]["generated_text"])
        df = pandas.DataFrame.from_dict(test_predictions)
        with mlflow.start_run(experiment_id=experiment):
            mlflow.set_tag('mlflow.runName', f"{name}")
            mlflow.log_params(strategy)
            mlflow.log_table(df, f"evaluation_{dataset_test._fingerprint}.json")

    del pipe, model, tokenizer
    torch.cuda.empty_cache()


@cli.command()
@click.option('-e', '--experiment', 'experiments', type=int, multiple=True, default=[])
@click.option('-r', '--run', 'selected_runs', type=str, multiple=True, default=[])
@click.option('--exclude', 'excluded_runs', type=str, multiple=True, default=[])
@click.option('-m',
              '--metrics',
              type=click.Choice(list(standard_metrics.keys()), case_sensitive=False),
              default=standard_metrics.keys(),
              multiple=True)
@click.option('--batch-size', type=int, default=12)
@click.option('--overwrite', type=bool, default=False, is_flag=True)
@click.option('--debug', type=bool, default=False)
def update_metrics(experiments, selected_runs, excluded_runs, metrics, batch_size, overwrite, debug):
    from src.mlflow_utils import get_run_list, mlflow, download_run_data
    import re
    import pandas as pd
    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    selected_runs += tuple(get_run_list(experiments))
    selected_runs = list(set(selected_runs))

    for run in excluded_runs:
        if run in selected_runs:
            selected_runs.remove(run)

    logging.info("selected runs: %s", selected_runs)
    logging.info("selected metrics: %s", metrics)

    with Progress() as progress:
        task = progress.add_task("Evaluating", total=len(selected_runs))
        for run in selected_runs:
            progress.update(task, description=f"Evaluating {run}", advance=0)

            artifacts = mlflow.artifacts.list_artifacts(run_id=run)
            for artifact in artifacts:
                if artifact_name := re.match(r"evaluation_[0-9a-z]*.json", artifact.path):
                    artifact_name = artifact_name.group(0)
                    evaluation_data = download_run_data(run, artifact_name)
                    if not evaluation_data:
                        raise RuntimeError(f"Could not download evaluation data for run {run}")

                    preds = pd.DataFrame(data=evaluation_data['data'], columns=evaluation_data['columns'])
                    df = run_metrics(preds.to_dict(orient="list"), metrics, batch_size, overwrite)

                    repository = get_artifact_repository(mlflow.get_run(run).info.artifact_uri)
                    repository.delete_artifacts(artifact_name)
                    with mlflow.start_run(run_id=run):
                        mlflow.log_table(df, artifact_name)

            progress.update(task, advance=1)

if __name__ == '__main__':
    cli()
