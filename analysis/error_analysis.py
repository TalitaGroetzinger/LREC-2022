"""A module for preparing the error analysis."""
import logging

import pandas as pd


logging.basicConfig(level=logging.INFO)


def get_errors(pred_file: str, data_file: str, relaxed=False):
    logging.info(f"file: {pred_file}")
    pred_df = pd.read_csv(pred_file, sep="\t")
    data_df = pd.read_csv(data_file, sep="\t", quoting=3)

    error_ids = []
    success_ids = []

    for _, row in pred_df.iterrows():
        if row["preds"] == row["gold"]:
            success_ids.append(row["ids"])
        else:
            if relaxed:
                if int(row["gold"]) == 1:
                    pass
                else:
                    error_ids.append(row["ids"])
            else:
                error_ids.append(row["ids"])

    logging.info(f"error_ids {len(error_ids)}")
    logging.info(f"success_ids {len(success_ids)}")

    find_errors(data_df, pred_df, error_ids, success_ids)


def find_errors(data_df, pred_df, error_ids, success_ids):
    errors = []
    success = []

    for _, row in data_df.iterrows():
        for filler_index in range(1, 6):
            composite_id = f"{row['Id']}_{filler_index}"
            pred_row = pred_df.query(f'ids == "{composite_id}"')

            try:
                prediction = pred_row["preds"].to_list()[0]
                label = pred_row["gold"].to_list()[0]
            except IndexError:
                continue

            repr = [
                row["Id"], row["Article title"], row["Section header"], row["Previous context"],
                row["Sentence"], row["Follow-up context"], row[f"Filler{filler_index}"],
                prediction, label
                ]

            if composite_id in error_ids:
                errors.append(repr)
            elif composite_id in success_ids:
                success.append(repr)
            else:
                logging.info(f"Row {row} with filler id {filler_index} not listed.")

    error_df = pd.DataFrame(errors, columns=["Id", "Article title", "Section header", "Previous context", "Sentence", "Follow-up context", "Filler", "Prediction", "Label"])
    success_df = pd.DataFrame(success, columns=["Id", "Article title", "Section header", "Previous context", "Sentence", "Follow-up context", "Filler", "Prediction", "Label"])

    error_df.to_csv("errors_relaxed.tsv", sep="\t")
    success_df.to_csv("success_relaxed.tsv", sep="\t")


if __name__ == "__main__":
    get_errors(
        pred_file="../data/perplexity-ranking-linear_7.tsv",
        data_file="../../captain_obvious_code/data/ClarificationTask_DevData_Oct22a.tsv",
        relaxed=True
    )
