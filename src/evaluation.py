import logging
import os
import sys

import click
from typing import List, Dict, Callable

import numpy as np
from evaluate import load
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

sys.path.insert(0, '../')

from src.mlflow_utils import mlflow, get_run_list


def placeholder():
    logging.error("NYI")

    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, float]:
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0
        }

    return calc_metric


def bertscore() -> Callable:
    metric = load("bertscore")

    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, float]:
        scores = metric.compute(predictions=pred, references=target, lang="de")
        return {
            'precision': scores['precision'][0],
            'recall': scores['recall'][0],
            'f1': scores['f1'][0]
        }

    return calc_metric


def sacrebleu() -> Callable:
    metric = load("sacrebleu")

    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, float]:
        scores = metric.compute(predictions=pred, references=target)
        return {
            'score': scores['score'],
        }

    return calc_metric


def bleurt() -> Callable:
    metric = load("bleurt")

    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, float]:
        scores = metric.compute(predictions=pred, references=target)['scores']
        return {
            'average': np.mean(scores),
            'median': np.median(scores),
            'stdev': np.std(scores)
        }

    return calc_metric


def meteor() -> Callable:
    metric = load("meteor")

    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, float]:
        scores = metric.compute(predictions=pred, references=target)
        return {
            'score': scores['meteor']
        }

    return calc_metric


# embedding_model="distilbert-base-german-cased"
def moverscore(embedding_model="distilbert-base-multilingual-cased") -> Callable:
    os.environ['MOVERSCORE_MODEL'] = embedding_model

    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, float]:
        from moverscore_v2 import word_mover_score
        from collections import defaultdict

        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)

        scores = word_mover_score(target, pred, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1,
                                remove_subwords=True)
        return {
            'average': np.mean(scores),
            'median': np.median(scores),
            'stdev': np.std(scores)
        }

    return calc_metric


def nist() -> Callable:
    metric = load("nist_mt")

    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, float]:
        scores = metric.compute(predictions=pred, references=target)
        return {
            'score': scores['nist_mt'],
        }

    return calc_metric


def rouge() -> Callable:
    metric = load("rouge")

    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, float]:
        scores = metric.compute(predictions=pred, references=target)
        return {
            'rouge1': scores['rouge1'],
            'rouge2': scores['rouge2'],
            'rougeL': scores['rougeL'],
            'rougeLsum': scores['rougeLsum']
        }

    return calc_metric


def sentence_embeddings() -> Callable:
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    # Initialize the evaluator
    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, float]:
        embedding_1 = model.encode(pred, convert_to_tensor=True)
        embedding_2 = model.encode(target, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(embedding_1, embedding_2).diagonal().cpu().numpy()
        return {
            'average': np.mean(scores),
            'median': np.median(scores),
            'stdev': np.std(scores)
        }

    return calc_metric


def ter() -> Callable:
    metric = load("ter")

    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, float]:
        scores = metric.compute(predictions=pred, references=target)
        return {
            'score': scores['score'],
        }

    return calc_metric


standard_metrics = {
    'BERTScore': bertscore,
    'SacreBLEU': sacrebleu,
    # 'BLEURT': bleurt,
    'METEOR': meteor,
    'Moverscore': moverscore,
    'NIST': nist,
    'ROUGE': rouge,
    'SentenceEmbeddings': sentence_embeddings,
    # 'TER': ter
}


@click.command()
@click.argument('test_set', type=click.Path(exists=True, file_okay=True, dir_okay=False),
                default='model_training/datasets/default.py')
@click.option('-m',
              '--metrics',
              type=click.Choice(list(standard_metrics.keys()), case_sensitive=False),
              default=standard_metrics.keys(),
              multiple=True)
@click.option('-e', '--experiment', 'experiments', type=int, multiple=True, default=[])
@click.option('-r', '--run', 'selected_runs', type=str, multiple=True, default=[])
@click.option('--debug', type=bool, default=True)
def evaluate(test_set, metrics, experiments, selected_runs, debug):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    selected_runs += tuple(get_run_list(experiments))
    logging.debug("selected runs: %s", selected_runs)
    logging.debug("selected metrics: %s", metrics)


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
        print(f"{name}: {metric()(pred, target)}")


if __name__ == '__main__':
    debug_metrics()
