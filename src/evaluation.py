from evaluate import load
from mlflow_setup import mlflow

bertscore = load("bertscore")

def _eval_bert_f1_score(predictions, targets):
    scores = bertscore.compute(predictions=predictions, references=targets, lang="de")
    return mlflow.metrics.MetricValue(
        scores=scores['f1']
    )


def _eval_sacrebleu(predictions, targets):
    scores = []
    for i in range(len(predictions)):
        if len(predictions[i]) > 10:
            scores.append("yes")
        else:
            scores.append("no")
    return mlflow.metrics.MetricValue(
        scores=scores,
        aggregate_results=standard_aggregations(scores),
    )

def _eval_anything(predictions, targets):
    scores = []
    for i in range(len(predictions)):
        if len(predictions[i]) > 10:
            scores.append("yes")
        else:
            scores.append("no")
    return mlflow.metrics.MetricValue(
        scores=scores,
        aggregate_results=standard_aggregations(scores),
    )


# Create an EvaluationMetric object.
bert_f1_score_metric = mlflow.metrics.make_metric(
    eval_fn=_eval_bert_f1_score, greater_is_better=True, name="BERT_F1_score"
)

sacrebleu_metric = mlflow.metrics.make_metric(
    eval_fn=_eval_sacrebleu, greater_is_better=True, name="SacreBLEU"
)