import logging
import os
import tempfile

import click
from typing import List, Dict, Callable

import numpy as np


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
    from evaluate import load
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
    from evaluate import load
    metric = load("sacrebleu")

    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, float]:
        scores = metric.compute(predictions=pred, references=target)
        return {
            'score': scores['score'],
        }

    return calc_metric


def bleurt() -> Callable:
    from evaluate import load
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
    from evaluate import load
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
    from evaluate import load
    metric = load("nist_mt")

    def calc_metric(pred: List[str], target: List[str]) -> Dict[str, float]:
        scores = metric.compute(predictions=pred, references=target)
        return {
            'score': scores['nist_mt'],
        }

    return calc_metric


def rouge() -> Callable:
    from evaluate import load
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
    from sentence_transformers import SentenceTransformer, util

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
    from evaluate import load
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
                default='src/model_training/datasets/default.py')
@click.option('-m',
              '--metrics',
              type=click.Choice(list(standard_metrics.keys()), case_sensitive=False),
              default=standard_metrics.keys(),
              multiple=True)
@click.option('-e', '--experiment', 'experiments', type=int, multiple=True, default=[])
@click.option('-r', '--run', 'selected_runs', type=str, multiple=True, default=[])
@click.option('--batch-size', type=int, default=128)
@click.option('--per-sample', type=bool, default=False)
@click.option("--seed", type=int, default=42)
@click.option("--shuffle", type=bool, default=True)
@click.option("--subset-test", type=float, default=-1)
@click.option('--debug', type=bool, default=True)
def evaluate(test_set, metrics, experiments, selected_runs, batch_size, per_sample, seed, shuffle, subset_test, debug):
    import torch
    from transformers import GenerationConfig, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from src.utils import import_module_from_path
    from src.mlflow_utils import mlflow, get_run_list, download_run_artifact

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    selected_runs += tuple(get_run_list(experiments))
    logging.debug("selected runs: %s", selected_runs)
    logging.debug("selected metrics: %s", metrics)

    run = selected_runs[0]

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
    dataset_module = import_module_from_path(test_set)

    dataset_module.DefaultTestSet.prompt_pattern = run_data.data.params["0_dataset_prompt_pattern"]

    if dataset_module.DefinitionDataset.extra_tokens:
        tokenizer.add_tokens(dataset_module.DefinitionDataset.extra_tokens)
    if dataset_module.DefinitionDataset.extra_special_tokens:
        tokenizer.add_tokens(dataset_module.DefinitionDataset.extra_special_tokens)
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
        logging.debug("Adapter '%s' loaded." % adapter_name)
    elif run_data.data.params["1_model_is_adapter"] == "False":
        # model needs to be reinstated
        with tempfile.TemporaryDirectory() as temp_path:
            weight_path = download_run_artifact(run, "model_data", temp_path)
            model.from_pretrained(weight_path)
        logging.debug("Fine-tuned Model loaded.")
    else:
        raise ValueError("Invalid adapter information")

    model.eval()
    model.to(device)

    dataset_test = dataset_module.DefaultTestSet.create_dataset(tokenizer, shuffle=shuffle, seed=seed,
                                                                subset_test=subset_test)

    def data():
        for item in dataset_test:
            yield item["debug_text"]

    pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=device)

    preds = []

    from tqdm.auto import tqdm
    for i, out in enumerate(tqdm(pipe(data(), batch_size=batch_size, max_length=50, num_beams=5, early_stopping=True), total=len(dataset_test))):
        preds += out["generated_text"]


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
    evaluate()
