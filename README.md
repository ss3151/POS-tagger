Cross-lingual transfer for POS tagging
==============================

Using cross-lingual transfer approach, we implement a Part-of-speech tagger 
for a low resource language.

Project Organization
------------
    ├── datasets/     <- Contains Huggingface datasets for the used languages.
    ├── predictions/  <- Contains stored predictions on the target-language test set.     
    ├── scripts/      <- Main project code. 
    ├── utils/        <- Utility files.
    ├── Pipfile       <- Pipfile for pipenv package manager.
    └──README.md      <- A README for using this project.
--------

### Environment setup

Use  `pipenv` ([docs](https://github.com/pypa/pipenv)) to manage the
virtual enviroment. (Check Pipfile)

```shell
pipenv install --dev
```

[comment]: <> (### Environment variables)

[comment]: <> (| Name                  | Default                                                   | Description                                                                  |)

[comment]: <> (| :---------------------| --------------------------------------------------------  | ---------------------------------------------------------------------------- |)

[comment]: <> (| `TOKENIZER_PATH`      | `data/external/nltk_data/tokenizers/punkt/slovene.pickle` | Optional. Path for loading a pickle for the NLTK `punkt` sentence tokenizer. |)


From the project root. Set up the virtual environment.
```
pipenv shell
```
To run the training script (example):
```
python train.py --model_name <MODEL_NAME_HUGGINGFACE>
--training_dir <PATH_TO_HUGGINGFACE_DATASET_TRAIN>
--test_dir <PATH_TO_HUGGINGFACE_DATASET_TEST>
--output_dir <PATH_TO_STORE_CHECKPOINTS>

```