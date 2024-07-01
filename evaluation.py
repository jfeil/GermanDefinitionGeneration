import gc
import logging
import os
import tempfile
from collections import defaultdict

import click
from typing import List, Dict, Callable

import pandas


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
            'nist_score': [scores]*len(target),
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
@click.option('--batch-size', type=int, default=32)
@click.option("--seed", type=int, default=42)
@click.option("--shuffle", type=bool, default=True)
@click.option("--subset-test", type=float, default=-1)
@click.option('--debug', type=bool, default=False)
def evaluate(test_set, metrics, experiments, selected_runs, batch_size, seed, shuffle, subset_test, debug):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from src.utils import import_module_from_path
    from src.mlflow_utils import mlflow, get_run_list, download_run_artifact

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    selected_runs += tuple(get_run_list(experiments))
    selected_runs = tuple(set(selected_runs))

    logging.info("selected runs: %s", selected_runs)
    logging.info("selected metrics: %s", metrics)

    for run in selected_runs:
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
            logging.info("Adapter '%s' loaded." % adapter_name)
        elif run_data.data.params["1_model_is_adapter"] == "False":
            # model needs to be reinstated
            with tempfile.TemporaryDirectory() as temp_path:
                weight_path = download_run_artifact(run, "model_data", temp_path)
                model.from_pretrained(weight_path)
            logging.info("Fine-tuned Model loaded.")
        else:
            raise ValueError("Invalid adapter information")

        model.eval()
        model.to(device)

        dataset_test = dataset_module.DefaultTestSet.create_dataset(tokenizer, shuffle=shuffle, seed=seed,
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
                yield item["debug_text"]

        from tqdm.auto import tqdm
        for out in tqdm(pipe(data_generator(), batch_size=batch_size, max_length=50, num_beams=5, early_stopping=True), total=len(dataset_test), desc="Inferencing"):
            assert len(out) == 1
            test_predictions['prediction'].append(out[0]["generated_text"])

        df = pandas.DataFrame.from_dict(test_predictions)

        del pipe, model, tokenizer
        torch.cuda.empty_cache()

        for metric in tqdm(metrics, desc="Evaluating"):
            metric_results = defaultdict(list)  # type: Dict[str, List[float]]
            cur_metric = standard_metrics[metric]()
            for i in tqdm(range(0, len(df), batch_size), leave=False, desc=metric):
                pred, target = test_predictions['prediction'][i:i+batch_size], test_predictions['gt'][i:i+batch_size]
                result = cur_metric.calc_metric(pred, target)
                for k, v in result.items():
                    metric_results[k] += v
            del cur_metric
            torch.cuda.empty_cache()
            gc.collect()
            for k, v in metric_results.items():
                df.insert(len(df.columns), k, v, False)
        with mlflow.start_run(run_id=run):
            mlflow.log_table(df, f"evaluation_{dataset_test._fingerprint}.json")
            logging.info(f"{run} data logged!")


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


if __name__ == '__main__':
    evaluate()
