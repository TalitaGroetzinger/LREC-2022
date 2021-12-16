from sent2vec.vectorizer import Vectorizer
import pdb

from transformers.utils.dummy_pt_objects import LineByLineWithRefDataset 
from data_preprocessing import merge_data
from helpers import train, evaluate
import
import torch
from torchtext.legacy import data
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn import svm
import numpy as np

from models import BERTClassification, SimpleBERT
SEED = 1234 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

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

def vectorizer(sentences): 
    """
        sentences: list with sentences 
    """
    
  
    vectorizer = Vectorizer()

    vectorizer.bert([sentences])
    vectors = vectorizer.vectors
    return vectors

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
        train_df.to_csv("./data/train.csv", index=False)
        development_df.to_csv("./data/dev.csv", index=False)

    
    Xtrain = train_df['text'].tolist()
    Ytrain = train_df['label'].tolist()
    Xdev = development_df['text'].tolist()
    Ydev = development_df['label'].tolist()

    return Xtrain, Ytrain, Xdev, Ydev 




def main():
    # read data and return buckets
    # at the moment, do not use the test data.
    Xtrain, Ytrain, Xdev, Ydev = read_data(use_context=True)

    Xdev_vec = np.array([vectorizer(sent) for sent in Xdev]).squeeze(1)
    Xtrain_vec =  np.array([vectorizer(sent) for sent in Xtrain]).squeeze(1)

    classifier = svm.LinearSVC()

    # (n_samples, n_features) -> (20, )
    classifier.fit(Xtrain_vec, Ytrain)
    print("done with fitting")
    
    Ydev_pred = classifier.predict(Xdev_vec).tolist()
    print("predicting")

    total_correct = 0 
    for pred, true_label in (Ydev_pred, Ydev):
        if pred == true_label: 
           total_correct +=1 
    
    accuracy_score = total_correct/len(Ydev)
    print("the accuracy is {0}".format(accuracy_score))
     
        






main()
