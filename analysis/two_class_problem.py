"""A module for transforming the plausibility classification into a two-class problem."""
import pandas as pd


def transform_into_binary_classification(score_file: str):
    score_df = pd.read_csv(score_file, sep="\t", quoting=3, names=["id", "score"])

    labels = []
    zero = 0
    one = 0

    for _, row in score_df.iterrows():
        if row["score"] <= 3:
            labels.append([row["id"], 0])
            zero += 1
        else:
            labels.append([row["id"], 1])
            one += 1

    print(f"zero {zero}")
    print(f"one {one}")

    label_df = pd.DataFrame(labels)
    label_df.to_csv("../data/ClarificationTask_DevScores_binary.tsv", sep="\t", index=False, header=None)



if __name__ == "__main__":
    transform_into_binary_classification(
        score_file="../../captain_obvious_code/data/ClarificationTask_DevScores_Dec12.tsv"
    )