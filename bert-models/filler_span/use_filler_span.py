"""A module for marking the filler span in the plausibility classification task."""
import random
import logging

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
from model import StartMarkerPlausibilityClassifier, DualInputPlausibilityClassifier, TwistedPlausibilityClassifier, SimplePlausibilityClassifier


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Dataset reading paths.
PathToTrainLabels = "../../data/ClarificationTask_TrainLabels_Sep23.tsv"
PathToTrainData = "../../data/ClarificationTask_TrainData_Sep23.tsv"
PathToDevLabels = "../../data/ClarificationTask_DevLabels_Dec12.tsv"
PathToDevData = "../../data/ClarificationTask_DevData_Oct22b.tsv"
PathToTestLabels = "../../data/ClarificationTask_TestLabels_Dec29.tsv"
PathToTestData = "../../data/ClarificationTask_TestData_Dec29a.tsv"


TESTING = True
LOAD_MODEL = "./models/bert_linear_vanilla.pt"

MODEL_NAME = "test_bert_linear_vanilla"
logging.basicConfig(filename=f"log_{MODEL_NAME}.log", level=logging.INFO)
logging.info(f"Log for {MODEL_NAME}")

logging.info("Hyperparameters")
NUM_FREEZED_LAYERS = 11
bert = BertModel.from_pretrained("bert-base-uncased")
freeze_bert_layers(bert=bert, num_layers=NUM_FREEZED_LAYERS)
logging.info(f"Freeze the first {NUM_FREEZED_LAYERS} bert layers") 

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
OUTPUT_DIM = 3
N_EPOCHS = 10
USE_CONTEXT = True
# FILLER_MARKERS = ("[F]", "[/F]")
FILLER_MARKERS = False
ADD_FILLER_MARKERS_TO_SPECIAL_TOKENS = False
LEARNING_RATE = 0.0001
CONSTRUCT_SENTENCE_PAIR = False
START_MARKER = False
TWISTED = False
DROPOUT = 0.5

logging.info(f"Testing: {TESTING}")
logging.info(f"Epochs: {N_EPOCHS}")
logging.info(f"Learning rate: {LEARNING_RATE}")
logging.info(f"Dropout rate: {DROPOUT}")
logging.info(f"Use context: {USE_CONTEXT}")
logging.info(f"Filler markers: {FILLER_MARKERS}")
logging.info(f"Add filler markers to special tokens: {ADD_FILLER_MARKERS_TO_SPECIAL_TOKENS}")
logging.info(f"Start marker: {START_MARKER}")
logging.info(f"Sentence pair: {CONSTRUCT_SENTENCE_PAIR}")
logging.info(f"Twisted: {TWISTED}")


def main():
    if TESTING:
        test()
    else:
        train_dev()


def test():
    logging.info("TEST")
    instance_transformation = InstanceTransformation(
        tokenizer=tokenizer,
        filler_markers=FILLER_MARKERS,
        use_context=USE_CONTEXT,
        construct_sentence_pair=CONSTRUCT_SENTENCE_PAIR,
    )
    batch_collator = BatchCollation(
        tokenizer=tokenizer, construct_sentence_pair=CONSTRUCT_SENTENCE_PAIR
    )
    test_dataset = PlausibilityDataset(
        instance_file=PathToTestData,
        label_file=PathToTestLabels,
        transformation=instance_transformation,
    )
    test_data_loader = get_data_loader(
        test_dataset, batch_size=16, collate_fn=batch_collator.collate
    )

    if START_MARKER and not TWISTED:
        logging.info("start marker model")
        model = StartMarkerPlausibilityClassifier(bert=bert, output_dim=OUTPUT_DIM, dropout=DROPOUT)
    elif START_MARKER and TWISTED:
        logging.info("twisted input model")
        model = TwistedPlausibilityClassifier(bert=bert, output_dim=OUTPUT_DIM, dropout=DROPOUT)
    else:
        logging.info("simple model")
        model = SimplePlausibilityClassifier(bert=bert, output_dim=OUTPUT_DIM, dropout=DROPOUT)

    # add filler markers to tokenizer vocabulary if necessary
    if FILLER_MARKERS and ADD_FILLER_MARKERS_TO_SPECIAL_TOKENS:
        tokenizer.add_special_tokens({"additional_special_tokens": FILLER_MARKERS})
        bert.resize_token_embeddings(len(tokenizer))

    logging.info(f"Loading model state dict from {LOAD_MODEL}")
    model.load_state_dict(torch.load(LOAD_MODEL, map_location=device))

    criterion = CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    _, test_acc = evaluate(model, test_data_loader, criterion, device, 0, MODEL_NAME)

    logging.info(f"Test Acc: {test_acc*100:.2f}%")
    print(f"Test Acc: {test_acc*100:.2f}%")


def train_dev():
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

    if START_MARKER and not TWISTED:
        logging.info("start marker model")
        model = StartMarkerPlausibilityClassifier(bert=bert, output_dim=OUTPUT_DIM, dropout=DROPOUT)
    elif START_MARKER and TWISTED:
        logging.info("twisted input model")
        model = TwistedPlausibilityClassifier(bert=bert, output_dim=OUTPUT_DIM, dropout=DROPOUT)
    else:
        logging.info("simple model")
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
        logging.info(f"Epoch {epoch}")
        print(f"Epoch {epoch}")
        train_loss, train_acc = train(
            model, train_data_loader, optimizer, criterion, device
        )
        valid_loss, valid_acc = evaluate(model, val_data_loader, criterion, device, epoch, MODEL_NAME)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{MODEL_NAME}.pt")

        logging.info("Epoch: {0}".format(epoch))
        logging.info(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        logging.info(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")


if __name__ == "__main__":
    main()
