"""A module for training and evaluating plausibility classifiers."""
import torch
import transformers


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return ((pred == target).sum() / len(pred)).float()


def freeze_bert_layers(bert: transformers.BertModel, num_layers: int) -> None:
    if num_layers < 0 or num_layers > 12:
        print(f"{num_layers} is not a valid number of bert layers in the range from 0 to 12.")

    modules = [bert.embeddings, *bert.encoder.layer[:num_layers]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def train(model, data_loader, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    num_batches = len(data_loader)

    for batch in data_loader:
        batch["text"] = batch["text"].to(device)
        batch["filler"] = batch["filler"].to(device)
        batch["filler_start_index"] = batch["filler_start_index"].to(device)
        batch["filler_end_index"] = batch["filler_end_index"].to(device)
        batch["label"] = batch["label"].to(device)

        logits = model(batch)
        loss = criterion(logits, batch["label"])
        acc = accuracy(logits=logits, target=batch["label"])

        loss.backward()
        optimizer.step()
        model.zero_grad()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / num_batches, epoch_acc / num_batches


def evaluate(model, data_loader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch in data_loader:
            batch["text"] = batch["text"].to(device)
            batch["filler"] = batch["filler"].to(device)
            batch["filler_start_index"] = batch["filler_start_index"].to(device)
            batch["filler_end_index"] = batch["filler_end_index"].to(device)
            batch["label"] = batch["label"].to(device)

            logits = model(batch)
            loss = criterion(logits, batch["label"])
            acc = accuracy(logits=logits, target=batch["label"])

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / num_batches, epoch_acc / num_batches
