# CommonLit Text Readability
#### PRJCTR task for Machine Learning in Production course



## How to reproduce training pipeline
The reproducibilaty of full pipeline takes approximately ~5-10 minutes

Step 1: create a virtual environment and activate it.
```
    python3.8 -m venv venv

    # linux/osx
    source venv/bin/activate

    # windows
    venv\Scripts\activate
```

Step 2: install requirements
```
    pip install -r requirements.txt
```

Step 3: load input data into directory `data/input/`
```
    mkdir data
    cd data

    gdown --folder https://drive.google.com/drive/u/0/folders/1oHRtTJa6Rl-yIhmvUsmMKPJuMp9dQrsm
    cd ..
```

Step 4: run bash script with all python scripts for pipeline
```
    chmod +x run_train_pipeline.sh
    ./run_train_pipeline.sh
```

In case you want look on all output files wihtour running pipeline, click on link below
https://drive.google.com/drive/u/0/folders/17y3XP6VEmd-02fpy4fhLCHcueHE3fgUk
