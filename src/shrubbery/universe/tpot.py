#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterable, List, Optional, Union

from numpy.typing import NDArray
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin
from tpot import TPOTRegressor

from shrubbery.utilities import PrintableModelMixin


class TPOTRegressorWrapper(
    BaseEstimator, MetaEstimatorMixin, RegressorMixin, PrintableModelMixin
):
    def __init__(
        self,
        search_space: str = 'linear',
        scorers: List[str] = ['neg_mean_squared_error'],
        scorers_weights: List[int] = [1],
        cv: Union[int, Iterable, Any] = 10,
        other_objective_functions: List = [],
        other_objective_functions_weights: List = [],
        objective_function_names: Optional[List] = None,
        bigger_is_better: bool = True,
        categorical_features: Optional[List] = None,
        memory: Any = None,
        preprocessing: Any = False,
        max_time_mins: float = 60.0,
        max_eval_time_mins: float = 10.0,
        n_jobs: int = 1,
        validation_strategy: str = 'none',
        validation_fraction: float = 0.2,
        early_stop: Optional[int] = None,
        warm_start: bool = False,
        periodic_checkpoint_folder: Optional[str] = None,
        verbose: int = 2,
        memory_limit: Optional[str] = None,
        client: Any = None,
        random_state: Optional[int] = None,
        allow_inner_regressors: Optional[bool] = None,
        population_size: int = 50,
    ) -> None:
        self.search_space = search_space
        self.scorers = scorers
        self.scorers_weights = scorers_weights
        self.cv = cv
        self.other_objective_functions = other_objective_functions
        self.other_objective_functions_weights = (
            other_objective_functions_weights
        )
        self.objective_function_names = objective_function_names
        self.bigger_is_better = bigger_is_better
        self.categorical_features = categorical_features
        self.memory = memory
        self.preprocessing = preprocessing
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
        self.n_jobs = n_jobs
        self.validation_strategy = validation_strategy
        self.validation_fraction = validation_fraction
        self.early_stop = early_stop
        self.warm_start = warm_start
        self.periodic_checkpoint_folder = periodic_checkpoint_folder
        self.verbose = verbose
        self.memory_limit = memory_limit
        self.client = client
        self.random_state = random_state
        self.allow_inner_regressors = allow_inner_regressors
        self.population_size = population_size

    def fit(
        self, x: NDArray, y: NDArray, **kwargs: Dict[str, Any]
    ) -> 'TPOTRegressorWrapper':
        estimator = TPOTRegressor(
            search_space=self.search_space,
            scorers=self.scorers,
            scorers_weights=self.scorers_weights,
            cv=self.cv,
            other_objective_functions=self.other_objective_functions,
            other_objective_functions_weights=(
                self.other_objective_functions_weights
            ),
            objective_function_names=self.objective_function_names,
            bigger_is_better=self.bigger_is_better,
            categorical_features=self.categorical_features,
            memory=self.memory,
            preprocessing=self.preprocessing,
            max_time_mins=self.max_time_mins,
            max_eval_time_mins=self.max_eval_time_mins,
            n_jobs=self.n_jobs,
            validation_strategy=self.validation_strategy,
            validation_fraction=self.validation_fraction,
            early_stop=self.early_stop,
            warm_start=self.warm_start,
            periodic_checkpoint_folder=self.periodic_checkpoint_folder,
            verbose=self.verbose,
            memory_limit=self.memory_limit,
            client=self.client,
            random_state=self.random_state,
            allow_inner_regressors=self.allow_inner_regressors,
            population_size=self.population_size,
        )
        estimator.fit(x, y)
        self.serialized_model_ = estimator.fitted_pipeline_
        return self.serialized_model_

    def predict(self, x: NDArray) -> NDArray:
        assert self.serialized_model_ is not None
        return self.serialized_model_.predict(x)
