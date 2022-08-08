from pathlib import Path
from typing import List

import hydra
import pandas as pd
import spacy
from omegaconf import DictConfig, OmegaConf
from spacy import glossary
from src.utils.helpers import check_create_folder, init_logger, save_json
from tqdm.auto import tqdm

LOGGER = init_logger('linguistic_parsing')

def extract_linguistic_features(text:str, nlp: spacy.lang):
    """
    Extract features from text

    First we extract basic one, like - text length, count of tokens, unique tokens, tokens per sentence, etc
    Then we extract counts and ratios - how many we have in text: verbs, nouns, punctuation, extracted entities, etc

    The main idea - discover how our text is grammaticaly/morphologically complex
    How difficult to read such grammar structure

    Args:
        text (str): input text
        nlp (spacy.lang): spacy model

    Returns:
        dict: extacted features
    """
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]
    text_length = len(text)
    count_of_tokens = len(tokens)
    count_unique_tokens = len(set(tokens))
    extracted_info = doc.to_json()
    count_sentences = len(extracted_info['sents'])
    
    tokens_pos = [token['pos'] for token in extracted_info['tokens']]
    tokens_tag = [token['tag'] for token in extracted_info['tokens']]
    tokens_dep = [token['dep'] for token in extracted_info['tokens']]
    ner_labels = [entity['label'] for entity in extracted_info['ents']]

    all_labels = tokens_pos + tokens_tag + tokens_dep + ner_labels

    extracted_features = {
        'text_length': text_length,
        'count_of_tokens': count_of_tokens,
        'count_unique_tokens': count_unique_tokens,
        'ratio_unique_tokens': count_unique_tokens/count_of_tokens,
        'count_sentences': count_sentences,
        'tokens_per_sentence':  count_of_tokens/count_sentences
    }

    for label in glossary.GLOSSARY.keys():
        label_count = all_labels.count(label)
        extracted_features['count_'+label+''] = label_count
        extracted_features['ratio_'+label+''] = label_count/count_of_tokens
    return extracted_features

def process_texts_linguistic(texts: List[str], spacy_model_name:str):
    """Extracting linguistic features for list of texts

    Args:
        texts (List[str]): texts to process
        spacy_model_name (str): name of corpus, on which spacy model was pretrained

    Returns:
        List[dict]: list of dicts with extracted features for each text
    """
    nlp = spacy.load(spacy_model_name)
    extracted_features = [extract_linguistic_features(text, nlp) for text in tqdm(texts)]
    return extracted_features

@hydra.main(version_base='1.2', config_path="../../configs/feature_generation", config_name="linguistic_parsing")
def main(conf: DictConfig):
    
    LOGGER.info('Config loaded: '+ str(OmegaConf.to_object(conf)))

    part = conf.data_part

    data_path = Path(conf.input.folder).joinpath(conf.input[f'filename'])
    df = pd.read_csv(data_path)
    LOGGER.info(f'The {part} part loaded: '+ str(df.shape))

    texts = df[conf.text_column]
    extracted_features = process_texts_linguistic(texts, conf.spacy_model_name)
    LOGGER.info(f'For {part} extracted {len(extracted_features)} features')

    check_create_folder(conf.output.folder)
    save_path = Path(conf.output.folder).joinpath(conf.output[f'filename'])
    
    save_json(save_path, extracted_features)
    LOGGER.info(f'The {part} features are saved in {save_path}')

if __name__ == "__main__":
    main()
