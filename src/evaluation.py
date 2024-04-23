from evaluate import load
import os

bertscore = load("bertscore")
rougescore = load("rouge")

def calc_bertscore(prediction, target):
    scores = bertscore.compute(predictions=prediction, references=target, lang="de")
    return [
        scores['precision'][0],
        scores['recall'][0],
        scores['f1'][0]
    ]


def calc_rougescore(prediction, target):
    scores = rougescore.compute(predictions=prediction, references=target)
    return [
        scores['rouge1'],
        scores['rouge2'],
        scores['rougeL'],
        scores['rougeLsum']
    ]


# embedding_model="distilbert-base-german-cased"
def calc_moverscore(prediction, target, embedding_model="distilbert-base-multilingual-cased"):
    os.environ['MOVERSCORE_MODEL'] = embedding_model
    
    from moverscore_v2 import get_idf_dict, word_mover_score 
    from collections import defaultdict
    
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    
    
    return word_mover_score(target, prediction, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
