#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin
from tpot import TPOTRegressor
from tpot.config import regressor_config_dict_light

from shrubbery.constants import COLUMN_INDEX_ERA
from shrubbery.data.ingest import locate_numerai_file
from shrubbery.observability import logger
from shrubbery.utilities import PrintableModelMixin

FEATURES_TPOT_CSV = 'features.tpot.csv'
DEFAULT_FEATURE_SET = 'default'


def tpot_override_regressor_config_dict_light() -> Dict:
    return regressor_config_dict_light.copy() | {
        'tpot.builtins.FeatureSetSelector': {
            'subset_list': [str(locate_numerai_file(FEATURES_TPOT_CSV))],
            'sel_subset': [DEFAULT_FEATURE_SET],
        }
    }


def _tpot_about(estimator: TPOTRegressor) -> None:
    with tempfile.NamedTemporaryFile(
        delete_on_close=False
    ) as temporary_handle:
        temporary_handle.close()
        pipeline_path = Path(temporary_handle.name)
        estimator.export(pipeline_path)
        with pipeline_path.open() as handle:
            logger.info(f'TPOT pipeline:\n{handle.read()}')


def _tpot_feature_sets_subset_list(features: List[int]) -> None:
    tpot_feature_sets = [
        {
            'Subset': DEFAULT_FEATURE_SET,
            'Size': len(features),
            'Features': ';'.join([str(feature) for feature in features]),
        }
    ]
    with open(locate_numerai_file(FEATURES_TPOT_CSV), 'w') as handle:
        pd.DataFrame(tpot_feature_sets).to_csv(
            handle, header=True, index=False
        )


class TPOTRegressorWrapper(
    BaseEstimator, MetaEstimatorMixin, RegressorMixin, PrintableModelMixin
):
    def __init__(
        self,
        generations: int = 100,
        population_size: int = 100,
        offspring_size: Optional[int] = None,
        mutation_rate: float = 0.9,
        crossover_rate: float = 0.1,
        scoring: Optional[Union[str, Callable]] = None,
        cv: Union[int, Iterable, Any] = 5,
        subsample: float = 1.0,
        n_jobs: int = 1,
        max_time_mins: Optional[int] = None,
        max_eval_time_mins: Optional[float] = 5,
        random_state: Optional[int] = None,
        config_dict: Optional[Dict] = None,
        template: Optional[str] = None,
        warm_start: bool = False,
        memory: Optional[str] = None,
        use_dask: bool = False,
        periodic_checkpoint_folder: Optional[str] = None,
        early_stop: Optional[int] = None,
        verbosity: int = 0,
        disable_update_check: bool = False,
        log_file: Optional[str] = None,
        features_offset: int = COLUMN_INDEX_ERA,
    ) -> None:
        self.generations = generations
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.scoring = scoring
        self.cv = cv
        self.subsample = subsample
        self.n_jobs = n_jobs
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
        self.random_state = random_state
        self.config_dict = config_dict
        self.template = template
        self.warm_start = warm_start
        self.memory = memory
        self.use_dask = use_dask
        self.periodic_checkpoint_folder = periodic_checkpoint_folder
        self.early_stop = early_stop
        self.verbosity = verbosity
        self.disable_update_check = disable_update_check
        self.log_file = log_file
        self.features_offset = features_offset

    def fit(
        self, x: NDArray, y: NDArray, **kwargs: Dict[str, Any]
    ) -> 'TPOTRegressorWrapper':
        _tpot_feature_sets_subset_list(
            list(range(x.shape[1]))[self.features_offset :]
        )
        estimator = TPOTRegressor(
            self.generations,
            self.population_size,
            self.offspring_size,
            self.mutation_rate,
            self.crossover_rate,
            self.scoring,
            self.cv,
            self.subsample,
            self.n_jobs,
            self.max_time_mins,
            self.max_eval_time_mins,
            self.random_state,
            self.config_dict,
            self.template,
            self.warm_start,
            self.memory,
            self.use_dask,
            self.periodic_checkpoint_folder,
            self.early_stop,
            self.verbosity,
            self.disable_update_check,
            self.log_file,
        )
        estimator.fit(x, y)
        _tpot_about(estimator)
        self.serialized_model_ = estimator.fitted_pipeline_
        return self

    def predict(self, x: NDArray) -> NDArray:
        assert self.serialized_model_ is not None
        return self.serialized_model_.predict(x)
