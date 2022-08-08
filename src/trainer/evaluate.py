from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_squared_error
from src.utils.helpers import check_create_folder, init_logger, save_json

LOGGER = init_logger('Evaluate')


@hydra.main(version_base='1.2', config_path="../../configs/trainer", config_name="evaluate")
def main(conf: DictConfig):
    
    LOGGER.info('Config loaded: '+ str(OmegaConf.to_object(conf)))

    y_true = load_target(conf, data_type = 'ground_truth')
    LOGGER.info('True target loaded: '+ str(y_true.shape))
    
    y_predicted = load_target(conf, data_type = 'prediction')
    LOGGER.info('Predicted target loaded: '+ str(y_predicted.shape))

    mse = mean_squared_error(y_true, y_predicted, squared=True)
    rmse = mean_squared_error(y_true, y_predicted, squared=False)
    metrics = {
        'mse': mse,
        'rmse': rmse
    }

    LOGGER.info('Metrics: '+ str(metrics))

    check_create_folder(conf.output.folder)

    save_path = Path(conf.output.folder).joinpath(conf.output.filename)
    save_json(save_path, metrics)
    LOGGER.info('Metrics saved to '+ str(save_path))

def load_target(conf: DictConfig, data_type: str):
    conf_dict = conf.input[data_type]
    path = Path(conf_dict['folder']).joinpath(conf_dict['filename'])
    df = pd.read_csv(path)
    y = df[conf_dict['target_column']]
    return y

if __name__ == "__main__":
    main()
