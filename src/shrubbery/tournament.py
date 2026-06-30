import time

import pandas as pd
import requests

from shrubbery.constants import (
    COLUMN_MMC,
    COLUMN_ROUND_NUMBER,
    COLUMN_V2_CORR20,
)
from shrubbery.data.ingest import locate_numerai_file
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
            logger.info('Submitting tournament predictions')
            napi.upload_predictions(
                file_path=str(prediction_path),
                model_id=model_id,
            )
            logger.info('Submitted tournament predictions')
            break
        except requests.exceptions.HTTPError as error:
            if (
                error.response is not None
                and error.response.status_code == 429
            ):
                logger.info('Backing off upload of tournament predictions')
                time.sleep(30 * 60)
            else:
                logger.exception('Network failure for tournament predictions')
                time.sleep(60)
        except Exception as error:
            logger.exception('Submission failure for tournament predictions')
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


def update_tournament_submissions(numerai_model_id: str) -> None:
    try:
        performances = get_performances(numerai_model_id)
    except KeyError:
        # New model, ignore error
        return
    performances[[COLUMN_ROUND_NUMBER, COLUMN_MMC, COLUMN_V2_CORR20]].to_csv(
        locate_numerai_file(f'parformances_{numerai_model_id}.csv')
    )
