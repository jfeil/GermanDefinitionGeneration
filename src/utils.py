from typing import List, Tuple, Dict

from tqdm.auto import tqdm

from src.mlflow_utils import get_run_list, download_run_data


class GT(dict):
    def __init__(self, prompt_data, ground_truth):
        super().__init__(self, title=prompt_data['title'], context_word=sanitize_context(prompt_data['context_word']),
                         context_sentence=prompt_data['context_sentence'], ground_truth=ground_truth)

    def __str__(self):
        return f"{self['context_word']} [{self['title']}] --- {self['context_sentence']} ===> {self['ground_truth']}"

    def __hash__(self):
        return hash((self['title'], self['context_word'], self['context_sentence'], self['ground_truth']))


def del_(inp: str, words: List[str]) -> str:
    for word in words:
        inp = inp.replace(word, "")
    return inp


def sanitize_context(context: str) -> str:
    return del_(context, [".", "?", "!", ":", "…", ";", "‚", "‘", "»", "«", "„", "'", "(",
                          "\"", "”", "“", "›", "‹"])


def sanitize_prediction(pred: str) -> str:
    return del_(pred, ['</s>', '<|eot_id|>'])


class ResponseDataset:
    def __init__(self, experiments: List[int] = None, runs: List[str] = None):
        if experiments is not None and type(experiments) is not list:
            raise ValueError('experiments must be a list of ints or None')
        if runs is not None and type(runs) is not list:
            raise ValueError('runs must be a list of strings or None')

        if experiments:
            runs = get_run_list(experiments)

        self._overview = {}  # type: Dict[GT, Dict[str, str]]

        for run in tqdm(runs):
            preds = download_run_data(run)
            for entry in preds['data']:
                x = GT(*entry[1:3])
                if x not in self._overview:
                    self._overview[x] = {}
                self._overview[x][run] = sanitize_prediction(entry[3])

    def search_title(self, title: str) -> List[Tuple[GT, Dict[str, str]]]:
        return_val = []
        for i in self._overview:
            if title != i["title"]:
                continue
            return_val.append((i, self._overview[i]))
        return return_val

    def export_label_studio(self):
        id_ = 0
        return_val = []
        for gt in self._overview:
            for run in self._overview[gt]:
                id_ += 1
                return_val.append(
                    {
                        "id": id_,
                        "data": {
                            "title": gt["title"],
                            "context_word": gt["context_word"],
                            "context_sentence": gt["context_sentence"],
                            "ground_truth": gt["ground_truth"],
                            "prediction": self._overview[gt][run],
                            "meta_info": {
                                "run_id": run
                            }
                        }
                    }
                )

        return return_val
