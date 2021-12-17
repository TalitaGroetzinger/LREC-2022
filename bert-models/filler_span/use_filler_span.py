"""A module for marking the filler span in the plausibility classification task."""
import random

import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertModel

from dataset import (
    BatchCollation,
    InstanceTransformation,
    PlausibilityDataset,
    get_data_loader,
)
from helpers import train, evaluate, freeze_bert_layers
from model import SimplePlausibilityClassifier, StartMarkerPlausibilityClassifier, LSTMStartMarkerPlausibilityClassifier


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Dataset reading paths.
PathToTrainLabels = "../../data/ClarificationTask_TrainLabels_Sep23.tsv"
PathToTrainData = "../../data/ClarificationTask_TrainData_Sep23.tsv"
PathToDevLabels = "../../data/ClarificationTask_DevLabels_Dec12.tsv"
PathToDevData = "../../data/ClarificationTask_DevData_Oct22a.tsv"

print("SETTINGS")
NUM_FREEZED_LAYERS = 12
bert = BertModel.from_pretrained("bert-base-uncased")
freeze_bert_layers(bert=bert, num_layers=NUM_FREEZED_LAYERS)
print(f"Freeze the first {NUM_FREEZED_LAYERS} bert layers") 

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
OUTPUT_DIM = 3
N_EPOCHS = 10
USE_CONTEXT = True
#FILLER_MARKERS = None
FILLER_MARKERS = ("[F]", "[/F]")
#FILLER_MARKERS = ("$", "$")
ADD_FILLER_MARKERS_TO_SPECIAL_TOKENS = True
LEARNING_RATE = 0.01
CONSTRUCT_SENTENCE_PAIR = False
START_MARKER = True
DROPOUT = 0.25
USE_LSTM = True
LSTM_LAYERS = 2
LSTM_HIDDEN_DIMENSION = 256

print(f"Epochs: {N_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Dropout rate: {DROPOUT}")
print(f"Use context: {USE_CONTEXT}")
print(f"Filler markers: {FILLER_MARKERS}")
print(f"Add filler markers to special tokens: {ADD_FILLER_MARKERS_TO_SPECIAL_TOKENS}")
print(f"Start marker: {START_MARKER}")
print(f"Sentence pair: {CONSTRUCT_SENTENCE_PAIR}")
print(f"LSTM: {USE_LSTM} -> layers {LSTM_LAYERS} and hidden dim {LSTM_HIDDEN_DIMENSION}")


def main():
    print("Start")
    instance_transformation = InstanceTransformation(
        tokenizer=tokenizer,
        filler_markers=FILLER_MARKERS,
        use_context=USE_CONTEXT,
        construct_sentence_pair=CONSTRUCT_SENTENCE_PAIR,
    )
    batch_collator = BatchCollation(
        tokenizer=tokenizer, construct_sentence_pair=CONSTRUCT_SENTENCE_PAIR
    )

    train_dataset = PlausibilityDataset(
        instance_file=PathToTrainData,
        label_file=PathToTrainLabels,
        transformation=instance_transformation,
    )
    train_data_loader = get_data_loader(
        train_dataset, batch_size=16, collate_fn=batch_collator.collate
    )

    val_dataset = PlausibilityDataset(
        instance_file=PathToDevData,
        label_file=PathToDevLabels,
        transformation=instance_transformation,
    )
    val_data_loader = get_data_loader(
        val_dataset, batch_size=16, collate_fn=batch_collator.collate
    )

    if START_MARKER:
        if USE_LSTM:
            model = LSTMStartMarkerPlausibilityClassifier(bert=bert, output_dim=OUTPUT_DIM, dropout=DROPOUT, lstm_layers=LSTM_LAYERS, lstm_hidden_dim=LSTM_HIDDEN_DIMENSION)
        else:
            model = StartMarkerPlausibilityClassifier(bert=bert, output_dim=OUTPUT_DIM, dropout=DROPOUT)
    else:
        model = SimplePlausibilityClassifier(bert=bert, output_dim=OUTPUT_DIM, dropout=DROPOUT)

    # add filler markers to tokenizer vocabulary if necessary
    if FILLER_MARKERS and ADD_FILLER_MARKERS_TO_SPECIAL_TOKENS:
        tokenizer.add_special_tokens({"additional_special_tokens": FILLER_MARKERS})
        bert.resize_token_embeddings(len(tokenizer))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch}")
        train_loss, train_acc = train(
            model, train_data_loader, optimizer, criterion, device
        )
        valid_loss, valid_acc = evaluate(model, val_data_loader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "../baseline-model.pt")

        print("Epoch: {0}".format(epoch))
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")


if __name__ == "__main__":
    main()
