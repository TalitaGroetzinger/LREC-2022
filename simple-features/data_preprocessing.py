from typing import List, Tuple
import pandas as pd
from torch._C import set_num_threads
import pdb 

def retrieve_instances_from_dataset(dataset, use_context: bool, filler_markers=None):
    """Retrieve sentences with insertions from dataset.

    :param dataset: dataframe with labeled data
    :param use_context: whether the context+title should be used or not
    :param filler_markers: optional tuple with start and end marker strs for filler span
    if this is empty, the filler span is not marked

    :return: a tuple with
    * a list of id strs
    * a list of sentence strs
    """
    # fill the empty values with empty strings
    dataset = dataset.fillna("")

    ids = []
    instances = []
    fillers = []
    contexts_before = []
    contexts_after = []
    sents_with_filler = []

    for _, row in dataset.iterrows():
        for filler_index in range(1, 6):
            ids.append(f"{row['Id']}_{filler_index}")

            if filler_markers:
                sent_with_filler = insert_filler_markers(
                    sentence=row["Sentence"],
                    filler=row[f"Filler{filler_index}"],
                    filler_markers=filler_markers,
                )
            else:
                sent_with_filler = row["Sentence"].replace(
                    "______", row[f"Filler{filler_index}"]
                )
            
            sents_with_filler.append(sent_with_filler)

            if use_context:
                if row["Follow-up context"]:
                    context = "{0} \n {1} \n{2} \n{3} \n{4}".format(
                        row["Article title"],
                        row["Section header"],
                        row["Previous context"],
                        sent_with_filler,
                        row["Follow-up context"],
                    )
                    context.replace("(...)", "")
                else:
                    context = "{0} \n {1} \n{2} \n{3}".format(
                        row["Article title"],
                        row["Section header"],
                        row["Previous context"],
                        sent_with_filler,
                    )
                    context.replace("(...)", "")

                instances.append(context)
              
            else:
                instances.append(sent_with_filler)

            fillers.append(row[f"Filler{filler_index}"])
            
            contexts_before.append(row["Previous context"])
            contexts_after.append(row["Follow-up context"])

    return ids, instances, fillers, contexts_before, contexts_after, sents_with_filler


def insert_filler_markers(
    sentence: str, filler: str, filler_markers: Tuple[str, str]
) -> str:
    """Insert marker token at the start and at the end of the filler span in the sentence.

    :param sentence: sentence str with "______" blank where the filler should be inserted
    :param filler: filler str
    :param filler_markers: tuple with start and end marker strs for filler span
    example: ("[F]", "[/F]") -> "This is a [F] really simple [/F] example."
    :return: sentence str with filler span surrounded by filler markers
    """

    filler_with_marker = "{0} {1} {2}".format(filler_markers[0], filler, filler_markers[1])
    sentence = sentence.replace("______", filler_with_marker)
    return sentence


def merge_data(
    path_to_instances, path_to_labels, use_context: bool, filler_markers=None
):
    """Merge the labels and instances together into one file.

    :param path_to_instances: path to tsv with instances from the dataset
    :param path_to_labels: path to tsv with labels (classes) form the dataset
    :param use_context: bool indicating if context should be used
    :param filler_markers: optional tuple with start and end marker strs for filler span
    if this is empty, the filler span is not marked
    :return: a dataframe with the following columns: ids, version, label
    """
    merged_df_dict = {"ids": [], "text": [], "fillers": [],  "label": [], "context_before": [], "context_after": [], "sents_with_filler": []}
    instances_df = pd.read_csv(path_to_instances, sep="\t")

    labels_df = pd.read_csv(path_to_labels, sep="\t", names=["Id", "Label"])
    label_dict = {row["Id"]: row["Label"] for _, row in labels_df.iterrows()}

    ids, instances, fillers, contexts_before, contexts_after, sents_with_filler = retrieve_instances_from_dataset(
        dataset=instances_df, use_context=use_context, filler_markers=filler_markers
    )

    for id_elem, instance, filler, context_before, context_after, sent_with_filler in zip(ids, instances, fillers, contexts_before, contexts_after, sents_with_filler):
        merged_df_dict["ids"].append(id_elem)
        merged_df_dict["text"].append(instance)
        merged_df_dict["fillers"].append(filler)
        merged_df_dict["context_before"].append(context_before)
        merged_df_dict["context_after"].append(context_after)
        merged_df_dict["sents_with_filler"].append(sent_with_filler)

        label = label_dict[id_elem]

        if label == "IMPLAUSIBLE":
            merged_df_dict["label"].append(0)
        elif label == "NEUTRAL":
            merged_df_dict["label"].append(1)
        elif label == "PLAUSIBLE":
            merged_df_dict["label"].append(2)
        else:
            raise ValueError(f"Label {label} is not a valid plausibility class.")

    merged_df = pd.DataFrame.from_dict(merged_df_dict)

    return merged_df
