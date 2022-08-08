from pathlib import Path
from typing import List

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from src.utils.helpers import check_create_folder, init_logger, save_np_array

LOGGER = init_logger('vectorization')


def vectorize_texts(texts: List[str], vectorization_model_name: str):
    """
    Encode texts into vectors using sentence transformer model
    Args:
        texts (List[str]): texts to encode
        vectorization_model_name (str): model name

    Returns:
        np.ndarray: encoded vectors
    """
    vectorization_model = SentenceTransformer(vectorization_model_name)
    vectors = vectorization_model.encode(texts, show_progress_bar=True)
    return vectors

@hydra.main(version_base='1.2', config_path="../../configs/feature_generation", config_name="vectorization")
def main(conf: DictConfig):
    
    LOGGER.info('Config loaded: '+ str(OmegaConf.to_object(conf)))

    part = conf.data_part

    data_path = Path(conf.input.folder).joinpath(conf.input['filename'])
    df = pd.read_csv(data_path)
    LOGGER.info(f'The {part} part loaded: '+ str(df.shape))

    texts = df[conf.text_column]
    vectors = vectorize_texts(texts, conf.vectorization_model_name)
    LOGGER.info(f'For {part} extracted {vectors.shape} vectors')

    check_create_folder(conf.output.folder)
    save_path = Path(conf.output.folder).joinpath(conf.output['filename'])
    
    save_np_array(save_path, vectors)
    LOGGER.info(f'The {part} vectors are saved in {save_path}')
        

if __name__ == "__main__":
    main()
