import os
import argparse
from collections import Counter
from typing import List

import numpy as np
from nptyping import ndarray
from datasets import load_metric, load_from_disk

from config import TAG2IDX


metric = load_metric('seqeval')


def load_predictions(filename: str) -> ndarray:
    with open(filename, 'rb') as f:
        predictions = np.load(f, allow_pickle=True)
        return predictions


def vote_predictions(pred1: ndarray,
                     pred2: ndarray,
                     pred3: ndarray) -> ndarray:
    """
    Decides on a prediction
    :param pred1: Stored predictions for the test set
    :param pred2: Stored predictions for the test set
    :param pred3: Stored predictions for the test set
    :return: Final predictions
    """
    final = []
    for mbert, cse, robi in zip(pred1, pred2, pred3):
        final_preds = []
        for mbert_tag, cse_tag, robi_tag in zip(mbert, cse, robi):
            x = Counter([mbert_tag, cse_tag, robi_tag]).most_common(1)[0][0]
            final_preds.append(x)
        final.append(final_preds)
    final = np.array(final, dtype=list)
    return final


def to_tags(_labels: List[List[int]]) -> List[List[str]]:
    idx2tag = {v: k for k, v in TAG2IDX.items()}
    true_tags = []
    for ls in _labels:
        tags_ls = []
        for elem in ls:
            tags_ls.append(idx2tag[elem])
        true_tags.append(tags_ls)
    return true_tags


def main(args):
    dataset_file = args.target_dataset
    results_dir = args.output_dir
    predicts1_file = args.predictions1
    predicts2_file = args.predictions2
    predicts3_file = args.predictions3

    predicts1 = load_predictions(predicts1_file)
    predicts2 = load_predictions(predicts2_file)
    predicts3 = load_predictions(predicts3_file)

    pred_final = vote_predictions(predicts1, predicts2, predicts3)

    dataset = load_from_disk(dataset_file)
    labels = dataset['test']['tags']
    true_tags = to_tags(labels)

    results = metric.compute(predictions=pred_final, references=true_tags)

    with open(os.path.join(results_dir, "eval_results.txt"), "w") as w:
        print(f"***** Eval results *****")
        for key, value in sorted(results.items()):
            w.write(f"{key} = {value}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--target_dataset", required=True, type=str)
    parser.add_argument("--output_dir", type=str, default="./out")
    parser.add_argument("--predictions1", type=str, required=True)
    parser.add_argument("--predictions2", type=str, required=True)
    parser.add_argument("--predictions3", type=str, required=True)

    arguments, _ = parser.parse_known_args()

    main(arguments)
