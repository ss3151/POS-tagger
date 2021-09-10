import os
import sys
import logging
import argparse
from typing import Optional

import numpy as np
import datasets

from datasets import load_from_disk, concatenate_datasets, load_metric
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments, DataCollatorForTokenClassification
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer

N_LABEL = 15
metric = load_metric("seqeval")


def main(args):
        
    # Hyper-parameters
    training_dir = args.training_dir
    second_dir = args.second_training_dir if args.second_training_dir else None
    test_dir = args.test_dir
    output_dir = args.output_dir
    output_data_dir = args.output_data_dir
    model_dir = args.model_dir

    model_name = args.model_name
    epochs = args.epochs
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    lr_rate = args.learning_rate

    target_size = args.few_shot_percentage
    
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.info(sys.argv)
    
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    tag2id = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CCONJ': 4, 'DET': 5,
              'INTJ': 6, 'NOUN': 7, 'NUM': 8, 'PART': 9, 'PRON': 10,
              'PROPN': 11, 'PUNCT': 12, 'SCONJ': 13, 'VERB': 14}

    idx2tag = {v: k for k, v in tag2id.items()}

    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def join_datasets(dataset1: DatasetDict,
                      dataset2: Optional[DatasetDict],
                      target: Optional[DatasetDict]) -> Dataset:
        """
        Concatenates all data splits. If two datasets are provided
        it creates joint-dataset.

        :param target:
        :param dataset1: Dataset of the source language for training
        :param dataset2: Dataset of the second source language for training
        :return: training Dataset
        """
        train = concatenate_datasets([dataset1["train"],
                                      dataset1["validation"],
                                      dataset1["test"]])
        if dataset2:
            train2 = concatenate_datasets([dataset2["train"],
                                          dataset2["validation"],
                                          dataset2["test"]])
            train = concatenate_datasets(train, train2)
        if target:
            train = concatenate_datasets(train, target)
        return train

    # Load dataset
    test_dataset = datasets.load_from_disk(test_dir)
    train_dataset = datasets.load_from_disk(training_dir)
    train_dataset2 = None
    subset = None
    if second_dir:
        train_dataset2 = datasets.load_from_disk(second_dir)
    if target_size > .0:
        subset = test_dataset['train'].train_test_split(train_size=target_size,
                                                        shuffle=True,
                                                        seed=1)
    train_dataset = join_datasets(train_dataset, train_dataset2, subset)

    print(train_dataset.num_rows)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")

    label_all_tokens = False

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"],
                                     truncation=True,
                                     is_split_into_words=True)

        _labels = []
        for ix, label in enumerate(examples["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=ix)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)

                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(
                        label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            _labels.append(label_ids)

        tokenized_inputs["labels"] = _labels
        return tokenized_inputs

    train_tokenized_set = train_dataset.map(tokenize_and_align_labels,
                                            batched=True)
    test_tokenized_set = test_dataset.map(tokenize_and_align_labels,
                                          batched=True)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    def compute_metrics(p):
        _predictions, _labels = p
        _predictions = np.argmax(_predictions, axis=2)

        # Remove ignored index (special tokens)
        _true_predictions = [
            [idx2tag[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(_predictions, _labels)
        ]
        _true_labels = [
            [idx2tag[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(_predictions, _labels)
        ]
        # cm = confusion_matrix(true_labels, true_predictions)
        _results = metric.compute(predictions=_true_predictions,
                                  references=_true_labels)
        return {
            "precision": _results["overall_precision"],
            "recall": _results["overall_recall"],
            "f1": _results["overall_f1"],
            "accuracy": _results["overall_accuracy"],
        }

    # download model from model hub
    model = AutoModelForTokenClassification.from_pretrained(model_name,
                                                            num_labels=N_LABEL)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        evaluation_strategy="epoch",
        logging_dir=f"{output_data_dir}/logs",
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        train_dataset=train_tokenized_set,
        eval_dataset=test_tokenized_set['validation'],
    )

    # train model
    if get_last_checkpoint(output_dir) is not None:
        logger.info("***** continue training *****")
        trainer.train(resume_from_checkpoint=output_dir)
    else:
        trainer.train()
    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset['validation'])

    with open(os.path.join(output_data_dir, "eval_results.txt"), "w") as w:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            w.write(f"{key} = {value}\n")

    #TODO: implement few-shot
    # subset = dataset_mk['train'].train_test_split(train_size=0.3,
    #                                               shuffle=True,
    #                                               seed=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters as command-line arguments to the script.
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning_rate", type=str, default=3e-5)
    parser.add_argument("--output_dir",
                        type=str, required=True)

    # Data and output directories
    parser.add_argument("--training_dir", type=str)
    parser.add_argument("--few_shot_percentage", type=float, default=.0)
    parser.add_argument("--second_training_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--output_data_dir", type=str)
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--n_gpus", type=str)

    arguments, _ = parser.parse_known_args()
    
    main(arguments)
