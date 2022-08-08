import json
import logging
import os
from typing import Any, Union

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(name)s:%(lineno)d} %(levelname)s - %(message)s",
)

def init_logger(name):
    return logging.getLogger(name)

def check_create_folder(folder_path: str):
    os.makedirs(folder_path, exist_ok = True)

def save_json(path: str, content: Union[list, dict]):
    with open(path, 'w') as f:
            json.dump(content, f)

def load_json(path: str):
    with open(path, 'r') as f:
        content = json.load(f)
    return content

def save_np_array(path: str, np_array: np.ndarray):
    np.save(path, np_array)

def load_np_array(path: str):
    np_array = np.load(path)
    return np_array

def save_model(path: str, model: Any):
    joblib.dump(model, path)

def load_model(path: str):
    model = joblib.load(path)
    return model

def save_parquet(path: str, df: pd.DataFrame):
    df.to_parquet(path, index=None)

def load_parquet(path: str):
    df = pd.read_parquet(path)
    return df
