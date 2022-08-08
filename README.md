# CommonLit Text Readability
#### PRJCTR task for Machine Learning in the Production course

This project aims to implement the NLP model, which identifies the appropriate reading level of a passage of text.
The project is inspired by [CommonLit Readability Prize](https://www.kaggle.com/c/commonlitreadabilityprize/overview) competition.

## Stage 1 - Implement NLP model

The baseline model - XGBoost Regressor, trained on extracted morphological features and encoded vectors from the text.

The main thoughts behind this implementation and further improvements:
- Does a passage of text have a complicated grammatical structure?
- Is it consist of commonly used words, or the majority of them are rare and topic-specific?
- Is the topic of the text commonly used?
- How difficult to predict the next word based on the previous words in the text?

### How to reproduce the training pipeline
The reproducibility of the whole pipeline takes approximately ~5-10 minutes.

Step 1: Create a virtual environment and activate it.
```
    python3.8 -m venv venv

    # linux/osx
    source venv/bin/activate

    # windows
    venv\Scripts\activate
```

Step 2: Install requirements
```
    pip install -r requirements.txt
```

Step 3: Load input data into directory `data/input/`
```
    mkdir data
    cd data

    gdown --folder https://drive.google.com/drive/u/0/folders/1oHRtTJa6Rl-yIhmvUsmMKPJuMp9dQrsm
    cd ..
```

Step 4: Run bash script with all python scripts for pipeline
```
    chmod +x run_train_pipeline.sh
    ./run_train_pipeline.sh
```

In case you want to look at all output files without running the pipeline, click on the [link](https://drive.google.com/drive/u/0/folders/17y3XP6VEmd-02fpy4fhLCHcueHE3fgUk) or download all files:
```
    gdown --folder https://drive.google.com/drive/u/0/folders/17y3XP6VEmd-02fpy4fhLCHcueHE3fgUk
```

## Stage 2 - Implement API server for the trained model from Stage 1

In progress

## Stage 3 - Deploy API from Stage 2 and make it publicly available

TODO
