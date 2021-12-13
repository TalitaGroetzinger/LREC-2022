from data_preprocessing import merge_data
from helpers import train, evaluate

import torch
from torchtext.legacy import data
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from models import BERTClassification, SimpleBERT
from feature_extraction import extract_features


bert = BertModel.from_pretrained("bert-base-uncased")

# Dataset reading paths.
PathToTrainLabels = "../data/ClarificationTask_TrainLabels_Sep23.tsv"
PathToTrainData = "../data/ClarificationTask_TrainData_Sep23.tsv"
PathToDevLabels = "../data/ClarificationTask_DevLabels_Dec12.tsv"
PathToDevData = "../data/ClarificationTask_DevData_Oct22a.tsv"

# Model parameters
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
MAX_SEQ_LEN = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 3
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
N_EPOCHS = 10
USE_CONTEXT = True
FILLER_MARKERS = None
ADD_FILLER_MARKERS_TO_SPECIAL_TOKENS = False
MODEL_NAME = "context-with-sep.pt"


# set sequential = False, those fields are not texts.
ids = data.Field()
label = data.Field(
    sequential=False, use_vocab=False, batch_first=True, dtype=torch.float
)
# label = data.LabelField(dtype=torch.float)
# set use_vocab=False to use own decoder
text = data.Field(
    use_vocab=False,
    tokenize=tokenizer.encode,
    lower=False,
    include_lengths=False,
    batch_first=True,
    fix_length=MAX_SEQ_LEN,
    pad_token=PAD_INDEX,
    unk_token=UNK_INDEX,
)
ids.build_vocab()
# label.build_vocab()
text.build_vocab()

fields = {"ids": ("ids", ids), "text": ("text", text), "label": ("label", label)}


def read_data(use_context):
    """
    :param: use_context: boolean indicating whether the text should contain the sentence+filler (use_context=False)
    or sentence+filler in context (use_context=True)
    :return:
    * train and development set in BucketIterator format.
    """
    if use_context:
        train_df = merge_data(
            path_to_instances=PathToTrainData,
            path_to_labels=PathToTrainLabels,
            filler_markers=FILLER_MARKERS,
            use_context=use_context,
        )
        development_df = merge_data(
            path_to_instances=PathToDevData,
            path_to_labels=PathToDevLabels,
            filler_markers=FILLER_MARKERS,
            use_context=use_context,
        )
        train_df.to_csv("./data/train.csv", index=False)
        development_df.to_csv("./data/dev.csv", index=False)

    else:
        train_df = merge_data(
            path_to_instances=PathToTrainData,
            path_to_labels=PathToTrainLabels,
            filler_markers=FILLER_MARKERS,
            use_context=use_context,
        )
        development_df = merge_data(
            path_to_instances=PathToDevData,
            path_to_labels=PathToDevLabels,
            filler_markers=FILLER_MARKERS,
            use_context=use_context,
        )
    
    print("extract features for train ..... ")
    train_with_features = extract_features(train_df, 'train_df_with_perplexity.tsv') 

    print("extract features for dev")
    dev_with_features = extract_features(development_df, 'dev_df_with_perplexity.tsv')
    return train_with_features, dev_with_features


def main():
    # read data and return buckets
    # at the moment, do not use the test data.
    read_data(use_context=USE_CONTEXT)



main()
