import itertools
import logging
import re
from collections import defaultdict

import click
import numpy as np
import pandas as pd
import pandasgui

from src.mlflow_utils import download_run_data


statistic_fns = {
    "Average": pd.Series.mean,
    "StdDev": pd.Series.std,
    "Median": pd.Series.median,
    "Max": pd.Series.max,
    "Min": pd.Series.min,
}


@click.command()
@click.option('-e', '--experiment', 'experiments', type=int, multiple=True, default=[])
@click.option('-r', '--run', 'selected_runs', type=str, multiple=True, default=[])
@click.option('--debug', type=bool, default=False)
def evaluate(experiments, selected_runs, debug):
    from src.mlflow_utils import mlflow, get_run_list, download_run_artifact

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    selected_runs += tuple(get_run_list(experiments))
    selected_runs = tuple(set(selected_runs))

    collected_data = defaultdict(lambda: pd.DataFrame(
        columns=pd.MultiIndex.from_tuples([], names=["Metric", "Statistics"])))

    for run in selected_runs:
        artifacts = mlflow.artifacts.list_artifacts(run_id=run)
        for artifact in artifacts:
            if artifact_name := re.match(r"evaluation_[0-9a-z]*.json", artifact.path):
                artifact_name = artifact_name.group(0)
                dataset_id = artifact_name.replace('evaluation_', '').replace('.json', '')
                evaluation_data = download_run_data(run, artifact_name)
                if not evaluation_data:
                    raise RuntimeError(f"Could not download evaluation data for run {run}")
                evaluation_data = pd.DataFrame(data=evaluation_data['data'], columns=evaluation_data['columns'])

                new_row = {}
                for column in evaluation_data.columns[5:]:
                    for k, v in statistic_fns.items():
                        new_row[(column, k)] = v(evaluation_data[column])
                    if column not in collected_data[dataset_id].columns.levels[0]:
                        collected_data[dataset_id][list(itertools.product([column], statistic_fns.keys()))] = np.NaN

                collected_data[dataset_id] = pd.concat([collected_data[dataset_id], pd.DataFrame([new_row], index=[run])])

    pandasgui.show(**collected_data)


if __name__ == '__main__':
    evaluate()
