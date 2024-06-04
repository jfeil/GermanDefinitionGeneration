import re
from typing import List, Tuple, Dict

from tqdm.auto import tqdm


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


def strip_(inp: str, words: List[str]) -> str:
    for word in words:
        inp = inp.strip(word)
    return inp


def sanitize_context(context: str) -> str:
    context = re.sub(r'<.*?>', '', context)
    return del_(context, ["?", "!", ":", ";", "‚", "‘", "»", "«", "„", "'",
                          "\"", "”", "“", "›", "‹", "[…]", "…"])


def sanitize_context_word(context) -> str:
    context = del_(context, ["[", "]", "<", ">", "–"])
    return strip_(context, [",", ".", "(", ")", "[", "]", "´", "/", "’", "¡", "¿", "′", "†"])


def sanitize_prediction(pred: str) -> str:
    return del_(pred, ['</s>', '<|eot_id|>'])
