"""A module for the evaluation metrics."""
from typing import List

import pandas as pd
from scipy.stats import spearmanr

from src.format_checker_for_submission import check_format_of_submission


def score_ranking_task(submission_file: str, truth_file: str) -> float:
    """Compute a performance score based on predictions for the ranking task.

    :param submission_file: str path to submission file with predictions
    :param truth_file: str path to file with ground truth labels
    :return: float Spearman's rank correlation coefficient
    """
    submission = pd.read_csv(submission_file, delimiter="\t")
    check_format_of_submission(submission, subtask="ranking")

    reference = pd.read_csv(truth_file, delimiter="\t")
    check_format_of_submission(reference, subtask="ranking")

    gold_ratings = []
    predicted_ratings = []

    for _, row in submission.iterrows():
        try:
            reference_indices = list(
                reference["Id"][reference["Id"] == row["Id"]].index
            )
        except KeyError:
            raise ValueError(
                f"Identifier {row['Id']} from submission not in reference file"
            )

        if len(reference_indices) > 1:
            raise ValueError(
                f"Identifier {row['Id']} appears several times in reference file"
            )
        else:
            reference_index = reference_indices[0]
            gold_ratings.append(float(reference["Rating"][reference_index]))
            predicted_ratings.append(float(row["Rating"]))

    return spearmans_rank_correlation(
        gold_ratings=gold_ratings, predicted_ratings=predicted_ratings
    )


def spearmans_rank_correlation(
    gold_ratings: List[float], predicted_ratings: List[float]
) -> float:
    """Score submission for the ranking task with Spearman's rank correlation.

    :param gold_ratings: list of float gold ratings
    :param predicted_ratings: list of float predicted ratings
    :return: float Spearman's rank correlation coefficient
    """
    if len(gold_ratings) == 1 and len(predicted_ratings) == 1:
        raise ValueError("Cannot compute rank correlation on only one prediction.")

    return spearmanr(a=gold_ratings, b=predicted_ratings)[0]
