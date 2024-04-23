from typing import List
import os

mlflow_env = {'MLFLOW_TRACKING_URI': 'http://localhost:5000', 'MLFLOW_S3_ENDPOINT_URL': 'http://localhost:9000', 'MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR': "False"}

for env in mlflow_env:
    os.environ[env] = mlflow_env[env]

import mlflow
import tempfile
import json

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_registry_uri("http://localhost:9001")

def get_run_list(experiment_ids: List[int]):
    runs = []
    for experiment_id in experiment_ids:
        runs += mlflow.search_runs(experiment_ids=str(experiment_id))['run_id'].values.tolist()
    return runs

def download_run_data(run_id: str):
    with tempfile.TemporaryDirectory() as temp_path:
        mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=temp_path)
        with open(os.path.join(temp_path, 'eval_results_table.json')) as file:
            return json.load(file)