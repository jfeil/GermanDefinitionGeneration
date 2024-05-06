from typing import List, Tuple

import os
import tempfile
import json

mlflow_env = {'MLFLOW_TRACKING_URI': 'http://localhost:5000', 'MLFLOW_S3_ENDPOINT_URL': 'http://localhost:9000', 'MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR': "False"}

for env in mlflow_env:
    os.environ[env] = mlflow_env[env]

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_registry_uri("http://localhost:9001")


def get_run_list(experiment_ids: List[int]):
    runs = []
    for experiment_id in experiment_ids:
        runs += mlflow.search_runs(experiment_ids=str(experiment_id))['run_id'].values.tolist()
    return runs


def download_run_data(run_id: str, file_name: str = 'eval_results_table.json'):
    with tempfile.TemporaryDirectory() as temp_path:
        mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=temp_path)
        if not os.path.exists(path := os.path.join(temp_path, file_name)):
            return None
        with open(path) as file:
            return json.load(file)


class Experiment:
    def __init__(self, system_prompts: Tuple[str], question_prompt: str, example_prompts: Tuple[Tuple[str, str, str]]):
        self.system_prompts = system_prompts
        self.question_prompt = question_prompt
        self.example_prompts = example_prompts

    def generate_examples(self):
        examples = []
        for context, word, meaning in self.example_prompts:
            examples += [
                {
                    "role": "user",
                    "content": self.question_prompt % (context, word)
                },
                {
                    "role": "assistant",
                    "content": meaning
                }
            ]
        return examples

    def __str__(self):
        return f"System prompts:\n{self.system_prompts}\n\nQuestion prompt:\n{self.question_prompt}\n\nExample prompts:\n{self.example_prompts}"

    def __repr__(self):
        return str(self)


def recreate_experiment(run_data):
    example_prompt = eval(run_data.data.params['example_prompt'])
    if len(example_prompt) > 0 and type(example_prompt[0]) == dict:
        example_prompt = [('Die Liebe überwindet alle Grenzen', 'Liebe', 'inniges Gefühl der Zuneigung für jemanden oder für etwas')]
    
    return Experiment(
        eval(run_data.data.params['system_prompt']),
        run_data.data.params['question_prompt'],
        example_prompt
    )