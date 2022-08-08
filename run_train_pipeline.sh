#!/bin/sh
python -m spacy download en_core_web_sm

python -m src.preparation.train_dev_split

python -m src.feature_generation.linguistic_parsing data_part=train
python -m src.feature_generation.linguistic_parsing data_part=dev

python -m src.feature_generation.vectorization data_part=train
python -m src.feature_generation.vectorization data_part=dev

python -m src.feature_generation.combine_features data_part=train
python -m src.feature_generation.combine_features data_part=dev

python -m src.trainer.train
python -m src.trainer.predict
python -m src.trainer.evaluate
