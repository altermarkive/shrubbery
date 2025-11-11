#!/usr/bin/env python3

import math
import time
from typing import Optional

import pandas as pd
import requests
import wandb

from shrubbery.constants import (
    COLUMN_MMC,
    COLUMN_ROUND_NUMBER,
    COLUMN_V2_CORR20,
)
from shrubbery.napi import napi
from shrubbery.observability import logger
from shrubbery.utilities import save_prediction


def submit_tournament_predictions(
    df: pd.DataFrame, numerai_model_id: str
) -> None:
    pred_col = df.columns.to_list()[0]
    prediction_path = save_prediction(df, f'tournament_{pred_col}')
    # Upload validation prediction (Submissions -> Models -> Upload Submission)
    model_id = napi.get_models()[numerai_model_id]
    while True:
        try:
            logger.info('Submitting prediction')
            napi.upload_predictions(
                file_path=str(prediction_path),
                model_id=model_id,
            )
            logger.info('Submitted prediction')
            if wandb.run is not None:
                tags = list(wandb.run.tags) if wandb.run.tags else []
                wandb.run.tags = tuple(tags + ['submitted'])
            break
        except requests.exceptions.HTTPError as error:
            if (
                error.response is not None
                and error.response.status_code == 429
            ):
                logger.info('Backing off')
                time.sleep(30 * 60)
            else:
                logger.exception('Network failure')
                time.sleep(60)
        except Exception as error:
            logger.exception('Submission failure')
            if 'Are you using the latest live ids' in str(error):
                break
            time.sleep(10)


def get_performances(numerai_model_id: str) -> pd.DataFrame:
    model_id = napi.get_models()[numerai_model_id]
    performances = pd.DataFrame(
        napi.round_model_performances_v2(model_id=model_id)
    )
    submission_scores = pd.json_normalize(
        performances['submissionScores'].apply(
            lambda scores: (
                {}
                if scores is None
                else {score['displayName']: score['value'] for score in scores}
            )
        )
    )
    return performances.join(submission_scores)


def get_projects():
    api = wandb.Api()
    return [project.name for project in api.projects()]


def update_tournament_submissions(numerai_model_id: str) -> None:
    try:
        performances = get_performances(numerai_model_id)
    except KeyError:
        # New model, ignore error
        return
    projects = get_projects()
    api = wandb.Api()
    for project in projects:
        runs = api.runs(project)
        for run in runs:
            if 'submitted' not in run.tags:
                continue
            if f'numerai_model_id:{numerai_model_id}' not in run.tags:
                continue
            if 'scored' in run.tags:
                continue
            round_number: Optional[int] = None
            for tag in run.tags:
                try:
                    round_number = int(tag)
                except ValueError:
                    continue
            if round_number is None:
                continue
            entry = performances[
                performances[COLUMN_ROUND_NUMBER] == round_number
            ]
            scores = {
                score_key: entry[score_key].item()
                for score_key in [COLUMN_MMC, COLUMN_V2_CORR20]
            }
            try:
                if any(math.isnan(value) for value in scores.values()):
                    continue
                run.summary.update(scores)
            except TypeError:
                # New model, ignore error
                return
            run.tags.append('scored')
            run.update()
