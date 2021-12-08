import pandas as pd 
import pdb 

def retrieve_instances_from_dataset(dataset, use_context): 
    """Retrieve sentences with insertions from dataset.

    :param dataset: dataframe with labeled data
    :param use_context: whether the context+title should be used or not 
    :return: a tuple with
    * a list of id strs
    * a list of sentence strs
    """
    # fill the empty values with empty strings
    dataset = dataset.fillna("")

    ids = []
    instances = []
    

    for _, row in dataset.iterrows():
        for filler_index in range(1, 6):
            ids.append(f"{row['Id']}_{filler_index}")

            sent_with_filler = row["Sentence"].replace(
                "______", row[f"Filler{filler_index}"]
            )

            if use_context: 
                if row['Follow-up context']: 
                    context = "{0} \n {1} \n{2} \n{3} \n{4}".format(row['Article title'],row['Section header'],row['Previous context'],sent_with_filler,row['Follow-up context'])
                    context.replace("(...)", '')
                else: 
                    context = "{0} \n {1} \n{2} \n{3}".format(row['Article title'],row['Section header'],row['Previous context'],sent_with_filler)
                    context.replace("(...)", '')
                instances.append(context)
            else: 
                instances.append(sent_with_filler)

    return ids, instances





def merge_data(path_to_instances, path_to_labels, use_context): 
    """Merge the labels and instances together into one file. 

    :param path_to_instances: path to tsv with instances from the dataset 
    :param path_to_labels: path to tsv with labels (classes) form the dataset 

    :return: a dataframe with the following columns: ids, version, label 
    """

    merged_df_dict = {"ids": [], "text": [], "label": []}
    instances_df = pd.read_csv(path_to_instances, sep='\t')

    labels_df = pd.read_csv(path_to_labels, sep='\t', names=["Id", "Label"])
    label_dict = {row["Id"]: row["Label"] for _, row in labels_df.iterrows()}
    
    ids, instances = retrieve_instances_from_dataset(instances_df, use_context)
    for id_elem, instance in zip(ids, instances): 
        merged_df_dict["ids"].append(id_elem)
        merged_df_dict["text"].append(instance)
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

