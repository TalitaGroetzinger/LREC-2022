"""A module for performing various analyses on the model predictions."""
import statistics

import pandas as pd


def analyse(pred_file: str):
    print(f"file: {pred_file}")
    pred_df = pd.read_csv(pred_file, sep="\t")

    id_to_preds = {}
    id_to_distribution = {}

    for _, row in pred_df.iterrows():
        instance_id = row["ids"].split("_")[0]
        if instance_id not in id_to_preds:
            id_to_preds[instance_id] = []

        id_to_preds[instance_id].append(row["preds"])

        if instance_id not in id_to_distribution:
            id_to_distribution[instance_id] = {index: 0 for index in range(3)}

        id_to_distribution[instance_id][int(row["preds"])] += 1

    num_identical = 0

    for id, preds in id_to_preds.items():
        if len(preds) != 5:
            print(f"{id}: {preds}")

        for pred in preds[1:]:
            if pred != preds[0]:
                break
        else:
            num_identical += 1
            print(f"{id}: {preds}")

    print(f"num identical: {num_identical}")

    count_implausible = []
    count_neutral = []
    count_plausible = []

    for id, distr in id_to_distribution.items():
        count_implausible.append(distr[0])
        count_neutral.append(distr[1])
        count_plausible.append(distr[2])

    for id, count in enumerate([count_implausible, count_neutral, count_plausible]):
        print(f"{id}: mean {statistics.mean(count)}")
        print(f"{id}: min {min(count)}")
        print(f"{id}: max {max(count)}")

    plausible_distribution = {index: 0 for index in range(6)}

    for num in count_plausible:
        plausible_distribution[num] += 1

    print(f"plausible distr: {plausible_distribution}")


if __name__ == "__main__":
    analyse(pred_file="../data/perplexity-ranking-linear_7.tsv")
