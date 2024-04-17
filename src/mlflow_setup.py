import os

mlflow_env = {'MLFLOW_TRACKING_URI': 'http://localhost:5000', 'MLFLOW_S3_ENDPOINT_URL': 'http://localhost:9000'}

for env in mlflow_env:
    os.environ[env] = mlflow_env[env]

import mlflow
from tqdm.notebook import tqdm

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_registry_uri("http://localhost:9001")

import mlflow