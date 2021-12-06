from transformers import BertTokenizer, BertForSequenceClassification
import pdb 
import torch 


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()

    for batch in iterator:
        
        
        optimizer.zero_grad()
        
        predictions = model(batch.version.to(device))
        label = batch.label.type(torch.LongTensor)
        label = label.to(device)
        loss = criterion(predictions, label)
        
        acc = categorical_accuracy(predictions, label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.version).squeeze(1)
            predictions = predictions.to(device)
            label = batch.label.type(torch.LongTensor)
            label = label.to(device)

            loss = criterion(predictions, batch.label)
            
            acc = categorical_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)