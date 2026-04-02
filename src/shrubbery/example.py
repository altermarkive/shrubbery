import os

from xgboost import XGBRegressor

from shrubbery.main import NumeraiRunner, config_content, main_arguments
from shrubbery.neutralization import NumeraiNeutralization

if __name__ == '__main__':
    arguments = main_arguments()
    NumeraiRunner(
        numerai_model_id=os.getenv('NUMERAI_MODEL', 'default'),
        notes='Example',
        version='latest',
        feature_set_name='small',  # fncv3_features
        retrain=True,
        adversarial_downsampling_ratio=None,
        estimator=NumeraiNeutralization(
            neutralization_cap=50,
            neutralization_proportion=1.0,
            neutralization_normalize=True,
            estimator=XGBRegressor(
                device='cuda',
                verbosity=1,
                n_jobs=-1,
            ),
        ),
    ).run(config_content(__file__), 'run_config.py')
