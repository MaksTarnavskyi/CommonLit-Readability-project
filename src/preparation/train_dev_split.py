import random
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from src.utils.helpers import check_create_folder, init_logger

LOGGER = init_logger('train_dev_split')

def split(df: pd.DataFrame, test_size: int, stratified: bool, target_column: str, random_state: int):
    """Split data on train/dev part

    Args:
        df (pd.DataFrame): dataframe to split
        test_size (int): dev_part ratio
        stratified (bool): save distribution of target value in the dev part
        target_column (str): column for statification
        random_state (int): fix random state for reproducibility

    Returns:
        pd.DataFrame, pd.DataFrame: train and dev dataframes
    """
    if stratified == True:
        sorted_df = df.sort_values(target_column)
        df_indexes = list(sorted_df.index)
        num_samples = sorted_df.shape[0]
        desired_num_dev_samples = round(num_samples * test_size)
        bin_size = round(num_samples/desired_num_dev_samples)

        dev_indexes = []
        for i in range(0, num_samples, bin_size):
            bin_indexes = df_indexes[i:i+bin_size]

            random.seed(random_state)
            sampled_indexes = random.sample(population = bin_indexes, k = 1)
            dev_indexes.extend(sampled_indexes)

        train_part = df.drop(dev_indexes, axis=0)
        dev_part = df.loc[dev_indexes, :]

    else:
        train_part, dev_part = train_test_split(df, test_size=test_size, random_state=random_state)
        
    return train_part, dev_part

@hydra.main(version_base='1.2', config_path="../../configs/preparation", config_name="train_dev_split")
def main(conf: DictConfig):
    """
    Perform stratified train/dev split based on continuous target value
    Save distribution of target values in dev set
    """
    # conf = OmegaConf.load('configs/preparation/train_dev_split.yaml')
    LOGGER.info('Config loaded: '+ str(OmegaConf.to_object(conf)))

    input_data_path = Path(conf.input.folder).joinpath(conf.input.filename)
    df = pd.read_csv(input_data_path)
    LOGGER.info('Data loaded, shape: '+ str(df.shape))

    train_df, dev_df = split(df, conf.test_size, conf.stratified, conf.target_column_name, conf.random_state)
    LOGGER.info(f'Data splited: train size - {str(train_df.shape)}, dev size: {str(dev_df.shape)}')

    check_create_folder(conf.output.folder)
 
    output_train_path = Path(conf.output.folder).joinpath(conf.output.train_filename)
    train_df.to_csv(output_train_path, index=None)
    LOGGER.info('Train data saved to '+str(output_train_path))
    
    output_dev_path = Path(conf.output.folder).joinpath(conf.output.dev_filename)
    dev_df.to_csv(output_dev_path, index=None)
    LOGGER.info('Dev data saved to '+str(output_dev_path))

if __name__ == "__main__":
    main()
