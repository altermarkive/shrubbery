import json
import sys

import pandas as pd
import wandb
from tqdm import tqdm

from shrubbery.constants import COLUMN_ROUND_NUMBER
from shrubbery.evaluation import METRIC_PREDICTION_ID, TABLE_EVALUATION
from shrubbery.tournament import get_performances, get_projects


def collect_from_project(project: str) -> list[dict]:
    api = wandb.Api()
    runs = api.runs(project)
    data = []
    for run in tqdm(runs):
        try:
            round_number = int(run.tags[0])
        except IndexError:
            continue
        except ValueError:
            continue
        for artifact in run.logged_artifacts():
            if TABLE_EVALUATION.replace(' ', '') in artifact.name:
                table = json.loads(
                    artifact.files()[0].download(replace=True).read()
                )
                columns = table['columns']
                columns.append(COLUMN_ROUND_NUMBER)
                for row in table['data']:
                    row.append(round_number)
                    data.append(dict(zip(columns, row)))
    return data


def collect_from_projects(projects: list[str]) -> pd.DataFrame:
    data = []
    for project in projects:
        part = collect_from_project(project)
        data.extend(part)
    df = pd.DataFrame(data)
    return df


def metric_correlation_to_true_contribution(
    numerai_model_id: str, performance_column: str
):
    projects = get_projects()
    collected = collect_from_projects(projects)
    performances = get_performances(numerai_model_id)
    performances_only = performances[[COLUMN_ROUND_NUMBER, performance_column]]
    results = collected.join(
        performances_only.set_index(COLUMN_ROUND_NUMBER),
        on=COLUMN_ROUND_NUMBER,
    )
    results = results[results[performance_column].notnull()]
    return (
        results.drop(columns=[COLUMN_ROUND_NUMBER, METRIC_PREDICTION_ID])
        .corr()[performance_column]
        .abs()
        .sort_values(ascending=False)
    )


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(
            'Usage: python collect_evaluation_tables.py <NUMERAI_MODEL> <PERFORMANCE_COLUMN>'  # noqa: E501
        )
        sys.exit(1)
    print(metric_correlation_to_true_contribution(sys.argv[1], sys.argv[2]))
