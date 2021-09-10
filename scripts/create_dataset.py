import os
import sys
import ast
import logging
import argparse
from typing import List, Tuple

import pyconll
import pandas as pd
from datasets import load_dataset
from datasets import ClassLabel, Value, Sequence


def main(args):

    training_file = args.train_file
    validation_file = args.validation_file
    test_file = args.test_file
    output_path = args.output_path
    
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.info(sys.argv)
    
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    def read_conll(filename: str) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Loads CONLLu file and extracts list of tokens and tags.
        :param filename: CONLLu file
        :return: Tokens and UPOS tags
        """
        file_conll = pyconll.load_from_file(filename)
        words = []
        tags = []
        for sentence in file_conll:
            w = []
            t = []
            for token in sentence:
                unused = ['X', 'SYM']
                if token.upos and token.upos not in unused:
                    w.append(token.form)
                    t.append(token.upos)
            words.append(w)
            tags.append(t)
        return words, tags

    train_words, train_tags = read_conll(training_file)
    dev_words, dev_tags = read_conll(validation_file)
    test_words, test_tags = read_conll(test_file)

    totals = train_tags + dev_tags + test_tags
    uni_tags = list(set(tag for sent_tag in totals for tag in sent_tag))
    uni_tags.sort()

    tag2id = {tag: idx for idx, tag in enumerate(uni_tags)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}

    def create_dataset(tokens: List[List[str]],
                       tags: List[List[str]], nameset: str):
        """
        Creates CSVs from lists of tokens and tags.

        :param tokens: List of tokens
        :param tags: List of UPOS tags
        :param nameset: Name of the stored dataframe
        """
        ints = [[tag2id[tag] for tag in sent_tag] for sent_tag in tags]
        df = pd.DataFrame(list(zip(tokens, ints)), columns=['tokens', 'tags'])
        df[["tokens", "tags"]] = df[["tokens", "tags"]].astype('O')
        df.to_csv(nameset, index=False)

    create_dataset(train_words, train_tags, 'train.csv')
    create_dataset(dev_words, dev_tags, 'dev.csv')
    create_dataset(test_words, test_tags, 'test.csv')

    dataset = load_dataset('csv', data_files={'train': f'train.csv',
                                              'validation': f'dev.csv',
                                              'test': f'test.csv'})
    
    new_features = dataset['train'].features.copy()
    new_features['tokens'] = Sequence(feature=Value(dtype='string', id=None))
    new_features['tags'] = Sequence(feature=ClassLabel(num_classes=len(tag2id),
                                                       names=list(uni_tags)))

    def process(ex):
        return {"tokens": ast.literal_eval(ex["tokens"]),
                "tags": ast.literal_eval(ex["tags"])}

    dataset = dataset.map(process, features=new_features)
    dataset.save_to_disk(output_path)

    logger.info(f"Dataset stored on %: {output_path}")

    os.remove("train.csv")
    os.remove("dev.csv")
    os.remove("test.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--output_path",
                        type=str,
                        required=True,
                        default='/datasets')
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)

    arguments, _ = parser.parse_known_args()
    
    main(arguments)
