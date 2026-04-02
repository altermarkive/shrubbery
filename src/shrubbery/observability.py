import logging
import warnings


def silence_false_positive_warnings() -> None:
    for message in [
        # XGBoost will handle CPU to GPU transfer of data
        '.*Falling back to prediction using DMatrix.*',
        # There is currently no way around LGBMRegressor naming features
        '.*but LGBMRegressor was fitted with feature names.*',
    ]:
        warnings.filterwarnings(
            'ignore', category=UserWarning, message=message
        )


def setup_logger() -> logging.Logger:
    logging_format = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=logging_format, level=logging.INFO)
    logging.getLogger('distributed.core').setLevel(logging.WARNING)
    logging.getLogger('distributed.scheduler').setLevel(logging.WARNING)
    logging.getLogger('distributed.nanny').setLevel(logging.WARNING)
    logging.getLogger('distributed.http.proxy').setLevel(logging.WARNING)
    logging.getLogger('cuml').setLevel(logging.INFO)
    logger = logging.getLogger('shrubbery')
    return logger


logger = setup_logger()
