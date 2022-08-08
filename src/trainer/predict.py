from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from src.utils.helpers import check_create_folder, init_logger, load_model, load_parquet

LOGGER = init_logger('Predict')


@hydra.main(version_base='1.2', config_path="../../configs/trainer", config_name="predict")
def main(conf: DictConfig):
    
    LOGGER.info('Config loaded: '+ str(OmegaConf.to_object(conf)))

    part = conf.data_part
    X_predict = load_features(conf)
    LOGGER.info(f'The {part} features loaded: '+ str(X_predict.shape))

    if conf.model_name == 'XGBRegressor':

        model_path = Path(conf.input.model.folder).joinpath(conf.input.model.filename)
        prediction_model = load_model(model_path)
        LOGGER.info(f'Model {conf.model_name} loaded')
        
        LOGGER.info(f'Model {conf.model_name} start prediction')
        y_predicted = prediction_model.predict(X_predict)

        y_predicted_df = pd.DataFrame({conf.output.target_column: y_predicted})

        check_create_folder(conf.output.folder)
        
        save_path = Path(conf.output.folder).joinpath(conf.output.filename)
        y_predicted_df.to_csv(save_path, index=None)

        LOGGER.info(f'Predictions saved to {save_path}')
    else:
        raise NotImplementedError('Unknown model')


def load_features(conf: DictConfig):
    data_path = Path(conf.input.data.folder).joinpath(conf.input.data.filename)
    df = load_parquet(data_path)
    return df


if __name__ == "__main__":
    main()
