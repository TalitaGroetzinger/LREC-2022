import torch


def categorical_accuracy(preds, y):
    """
    Compute the accuracy for multi-class classification by taking the argmax.

    :param preds: predictions provided by the model
    :param y: the gold labels
    :return: the accuracy score for a batch.
    """
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device, USE_RANK):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        if USE_RANK: 
            ranking = batch.rank.type(torch.LongTensor)
        else: 
            ranking = batch.perplexity
        predictions = model(batch.text.to(device), ranking.to(device)).squeeze(1)

        label = batch.label.type(torch.LongTensor)
        label = label.to(device)
        loss = criterion(predictions, label)

        acc = categorical_accuracy(predictions, label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device, USE_RANK):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:
            if USE_RANK: 
                ranking = batch.rank.type(torch.LongTensor)
            else: 
                ranking = batch.perplexity

            predictions = model(batch.text.to(device), ranking.to(device)).squeeze(1)
            predictions = predictions.to(device)
            label = batch.label.type(torch.LongTensor)
            label = label.to(device)

            loss = criterion(predictions, label)

            acc = categorical_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
