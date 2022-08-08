from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from src.utils.helpers import check_create_folder, init_logger, load_parquet, save_model
from xgboost import XGBRegressor

LOGGER = init_logger('Trainer')


def fit_model(X_train: pd.DataFrame, y_train: pd.Series, model_params: dict):
    model = XGBRegressor(**model_params)
    model.fit(X_train, y_train)
    return model


@hydra.main(version_base='1.2', config_path="../../configs/trainer", config_name="train")
def main(conf: DictConfig):
    
    LOGGER.info('Config loaded: '+ str(OmegaConf.to_object(conf)))

    X_train = load_features(conf)
    LOGGER.info(f'Train features loaded: '+ str(X_train.shape))

    y_train = load_target(conf)
    LOGGER.info(f'Train target loaded: '+ str(y_train.shape))

    if conf.model_name == 'XGBRegressor':

        LOGGER.info(f'Start fittting model {conf.model_name} with params {str(conf.model_params)}')
        model = fit_model(X_train, y_train, model_params = conf.model_params)
        LOGGER.info(f'Model {conf.model_name} fitted')

        check_create_folder(conf.output.folder)
        
        save_path = Path(conf.output.folder).joinpath(conf.output.model_filename)
        save_model(save_path, model)
        LOGGER.info(f'Model saved to {save_path}')
    else:
        raise NotImplementedError('Unknown model')


def load_target(conf: DictConfig):
    data_path = Path(conf.input.folder).joinpath(conf.input.split_filename)
    df = pd.read_csv(data_path)
    target = df[conf.target_column]
    return target

def load_features(conf: DictConfig):
    data_path = Path(conf.input.folder).joinpath(conf.input.features_filename)
    df = load_parquet(data_path)
    return df


if __name__ == "__main__":
    main()
