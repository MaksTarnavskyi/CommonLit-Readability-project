from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from src.utils.helpers import (
    check_create_folder,
    init_logger,
    load_json,
    load_np_array,
    save_parquet,
)

LOGGER = init_logger('Combine features')


@hydra.main(version_base='1.2', config_path="../../configs/feature_generation", config_name="combine_features")
def main(conf: DictConfig):
    
    LOGGER.info('Config loaded: '+ str(OmegaConf.to_object(conf)))

    part = conf.data_part

    features_df = load_features_df(conf)
    LOGGER.info(f'The {part} features loaded: '+ str(features_df.shape))

    vectors_df = load_vectors_df(conf)
    LOGGER.info(f'The {part} vectors loaded: '+ str(vectors_df.shape))

    combined_df = pd.concat((features_df, vectors_df), axis=1)
    LOGGER.info(f'The {part} dataset prepeared: '+ str(combined_df.shape))

    check_create_folder(conf.output.folder)

    save_path = Path(conf.output.folder).joinpath(conf.output.filename)
    save_parquet(save_path, combined_df)
    LOGGER.info(f'The {part} combined features are saved in {save_path}')



def load_features_df(conf: DictConfig):
    features_path = Path(conf.input.folder).joinpath(conf.input.linguistic_features_filename)
    features = load_json(features_path)
    features_df = pd.DataFrame(features)
    return features_df

def load_vectors_df(conf: DictConfig):
    vectors_path = Path(conf.input.folder).joinpath(conf.input.text_vectors_filename)
    vectors = load_np_array(vectors_path)
    vectors_column_names = ['vec_'+str(i) for i in range(vectors.shape[1])]
    vectors_df = pd.DataFrame(vectors, columns = vectors_column_names)
    return vectors_df

def save_combined_df(conf: DictConfig, df: pd.DataFrame):
    save_path = Path(conf.output.folder).joinpath(conf.output.filename)
    df.to_parquet(save_path, index=None)
    

if __name__ == "__main__":
    main()
