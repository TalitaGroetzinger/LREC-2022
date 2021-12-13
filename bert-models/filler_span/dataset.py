from typing import Dict, List, Tuple

import pandas as pd
import torch
import transformers


class BatchCollation:
    def __init__(self, tokenizer: transformers.BertTokenizer):
        self.tokenizer = tokenizer

    def collate(self, batch: List[Dict]):
        """ """
        result = {}

        keys = ["identifier", "text", "filler_start_index", "filler_end_index", "label"]

        for key in keys:
            result[key] = []

        for instance_dict in batch:
            for key in keys:
                result[key].append(instance_dict[key])

        for key in keys:
            if key not in ["identifier", "text"]:
                result[key] = torch.tensor(result[key])

        result["text"] = self.tokenizer(
            text=result["text"], return_tensors="pt", padding=True
        )

        return result


class InstanceTransformation:
    def __init__(self, tokenizer, filler_markers=None, use_context=False):
        self.tokenizer = tokenizer
        self.filler_markers = filler_markers
        self.use_context = use_context

    def transform(
        self,
        identifier: str,
        text: str,
        filler: str,
        label: int,
        title: str,
        section_header: str,
        previous_context: str,
        follow_up_context: str,
    ) -> Dict:
        start_index = None
        end_index = None

        if self.filler_markers:
            sent_with_filler, start_index, end_index = self.insert_filler_markers(
                sentence=text,
                filler=filler,
                filler_markers=self.filler_markers,
            )
        else:
            try:
                sent_with_filler = text.replace("______", filler)
            except ValueError:
                raise ValueError(f"Sentence {text} does not contain blank.")

        if self.use_context:
            if follow_up_context:
                context = f"{title} {section_header} {previous_context} {sent_with_filler} {follow_up_context}"
            else:
                context = (
                    f"{title} {section_header} {previous_context} {sent_with_filler}"
                )

            context = context.replace("(...)", "")
            text = context
        else:
            text = sent_with_filler

        return {
            "identifier": identifier,
            "text": text.replace("______", filler),
            "label": label,
            "filler_start_index": start_index,
            "filler_end_index": end_index,
        }

    def insert_filler_markers(
        self, sentence: str, filler: str, filler_markers: Tuple[str, str]
    ) -> Tuple[str, int, int]:
        """Insert marker token at the start and at the end of the filler span in the sentence.

        :param sentence: sentence str with "______" blank where the filler should be inserted
        :param filler: filler str
        :param filler_markers: tuple with start and end marker strs for filler span
        example: ("[F]", "[/F]") -> "This is a [F] really simple [/F] example."
        :return: a tuple with
        * sentence str with filler span surrounded by filler markers
        * int start index of the filler span
        * int end index of the filler span
        """
        if "______" not in sentence:
            raise ValueError(f"Sentence {sentence} does not contain blank.")

        insertion = f" {filler_markers[0]} {filler} {filler_markers[1]} "
        filler_span_len = len(self.tokenizer.tokenize(insertion))

        tokens = sentence.split()
        punct_before = False
        punct_after = False
        blank_index = 0

        for index, token in enumerate(tokens):
            if "______" == token:
                blank_index = index
                break
            elif "______" in token:
                blank_index = index

                subtokens = token.split("______")
                if len(subtokens) == 2:
                    if not subtokens[0]:
                        punct_after = True
                    if not subtokens[1]:
                        punct_before = True
                    if subtokens[0] and subtokens[1]:
                        punct_before = True
                        punct_after = True
                else:
                    raise ValueError(f"Token {token} has odd format for a blank.")
                break

        if punct_before or punct_after:
            tokens = (
                tokens[:blank_index]
                + [
                    subtoken
                    for subtoken in tokens[blank_index].split("______")
                    if subtoken
                ]
                + tokens[blank_index + 1 :]
            )

        if punct_before:
            tokens_before = tokens[: blank_index + 1]

        else:
            tokens_before = tokens[:blank_index]

        tokenized_before = self.tokenizer.tokenize(" ".join(tokens_before))
        sentence_before_len = len(tokenized_before)

        new_sentence = sentence.replace("______", insertion)
        start_index = sentence_before_len
        end_index = start_index + filler_span_len - 1

        return new_sentence, start_index, end_index


class PlausibilityDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        instance_file: str,
        label_file: str,
        transformation: InstanceTransformation,
    ):
        super().__init__()
        self.transformation = transformation

        instances = pd.read_csv(instance_file, sep="\t", quoting=3)
        instances.fillna("")

        labels = pd.read_csv(
            label_file, sep="\t", quoting=3, header=None, names=["Id", "Label"]
        )
        transformed_labels = [transform_label(label) for label in labels["Label"]]

        self.data = []

        for _, row in instances.iterrows():
            for filler_index in range(1, 6):
                self.data.append(
                    {
                        "identifier": f"{row['Id']}_{filler_index}",
                        "text": row["Sentence"],
                        "filler": row[f"Filler{filler_index}"],
                        "title": row["Article title"],
                        "section_header": row["Section header"],
                        "previous_context": row["Previous context"],
                        "follow_up_context": row["Follow-up context"],
                    }
                )

        assert len(transformed_labels) == len(self.data)

        for instance_dict, label in zip(self.data, transformed_labels):
            instance_dict["label"] = label

    def __getitem__(self, item: int):
        return self.transformation.transform(**self.data[item])

    def __len__(self):
        return len(self.data)


def get_data_loader(dataset: PlausibilityDataset, batch_size: int, collate_fn):
    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )


def collate_batch(batch: List[Dict]) -> Dict:
    result = {}

    keys = ["identifier", "text", "filler_start_index", "filler_end_index", "label"]

    for key in keys:
        result[key] = []

    for instance_dict in batch:
        for key in keys:
            result[key].append(instance_dict[key])

    for key in keys:
        if key not in ["identifier", "text"]:
            result[key] = torch.tensor(result[key])

    for sent in result["text"]:
        pass

    return result


def transform_label(label: str) -> int:
    if label == "IMPLAUSIBLE":
        return 0
    elif label == "NEUTRAL":
        return 1
    elif label == "PLAUSIBLE":
        return 2
    else:
        raise ValueError(f"Label {label} is not a valid plausibility class.")


if __name__ == "__main__":
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    instance_transformation = InstanceTransformation(
        tokenizer=tokenizer, filler_markers=("$", "$")
    )
    batch_collator = BatchCollation(tokenizer=tokenizer)

    dataset = PlausibilityDataset(
        instance_file="../../../captain_obvious_code/data/ClarificationTask_TrainData_Sep23.tsv",
        label_file="../../../captain_obvious_code/data/ClarificationTask_TrainLabels_Sep23.tsv",
        transformation=instance_transformation,
    )

    data_loader = get_data_loader(
        dataset, batch_size=16, collate_fn=batch_collator.collate
    )
    for i, instance in enumerate(data_loader):
        print(instance)
