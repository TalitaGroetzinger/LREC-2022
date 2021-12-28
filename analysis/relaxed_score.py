"""A module for performing various analyses on the model predictions."""
import logging
from typing import List

import pandas as pd


logging.basicConfig(level=logging.INFO)


def relax(pred_file: str):
    logging.info(f"file: {pred_file}")
    pred_df = pd.read_csv(pred_file, sep="\t")

    predictions = list(pred_df["preds"])
    labels = list(pred_df["gold"])

    strict_true_positives = 0

    for prediction, label in zip(predictions, labels):
        if prediction == label:
            strict_true_positives += 1

    logging.info(f"len old gold {len(predictions)}")
    logging.info(f"old true positives {strict_true_positives}")
    logging.info(f"old accuracy {strict_true_positives / len(predictions)}")

    new_pred = []
    new_gold = []
    true_positives = 0

    for prediction, label in zip(predictions, labels):
        if label == 0:
            continue

        new_pred.append(prediction)
        new_gold.append(label)

        if prediction == label:
            true_positives += 1

    logging.info(f"len new gold {len(new_gold)}")
    logging.info(f"new true positives {true_positives}")
    logging.info(f"new accuracy {true_positives / len(new_gold)}")

    for i in range(3):
        res = class_specific_accuracy(predictions, labels, class_indices=[i])
        logging.info(f"class {i}: {res}")

    res = class_specific_accuracy(predictions, labels, class_indices=[0, 2])
    logging.info(f"without 1: {res}")

    res = class_specific_accuracy(predictions, labels, class_indices=[0, 1, 2])
    logging.info(f"all: {res}")


def class_specific_accuracy(predictions, labels, class_indices: List[int]) -> float:
    true_positives = 0
    total = 0

    for prediction, label in zip(predictions, labels):
        if label in class_indices:
            total += 1
            if prediction == label:
                true_positives += 1

    logging.info(f"true positives {true_positives} / {total}")
    return true_positives / total

if __name__ == "__main__":
    relax(pred_file="../data/perplexity-ranking-linear_7.tsv")

