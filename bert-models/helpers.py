import torch
import pandas as pd 

def categorical_accuracy(preds, y, ids, eval):
    """
    Compute the accuracy for multi-class classification by taking the argmax.

    :param preds: predictions provided by the model
    :param y: the gold labels
    :param ids: batch.ids RawField()
    :param eval: indicates whether evaluation mode should be on or not. If true then the predcition file will be made. 
    :return: the accuracy score for a batch.
    """
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    predictions_list = [pred[0] for pred in top_pred.tolist()]
    y_gold = y.tolist()
    if eval: 
        instance_ids_for_tsv_file = []
        preds_for_tsv_file = []
        y_gold_for_tsv_file = []
        for instance_id, pred, y_gold in zip(ids, predictions_list, y_gold):
            preds_for_tsv_file.append(pred)
            y_gold_for_tsv_file.append(y_gold) 
            instance_ids_for_tsv_file.append(instance_id)
        return acc, instance_ids_for_tsv_file, y_gold_for_tsv_file, preds_for_tsv_file
    else: 
        return acc 


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text.to(device))
        label = batch.label.type(torch.LongTensor)
        label = label.to(device)
        loss = criterion(predictions, label)

        acc = categorical_accuracy(predictions, label, batch.ids, eval=False)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device, epoch_nr, model_name):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    df_for_evaluation_dict = {"ids": [], "preds": [], "gold": []}
    with torch.no_grad():

        for batch in iterator:

            predictions = model(batch.text.to(device))
            label = batch.label.type(torch.LongTensor)
            label = label.to(device)
            loss = criterion(predictions, label)

            acc, instance_ids_for_tsv_file, y_gold_for_tsv_file, preds_for_tsv_file  = categorical_accuracy(predictions, label, batch.ids, eval=True)
            df_for_evaluation_dict['ids'] += instance_ids_for_tsv_file
            df_for_evaluation_dict['preds'] += preds_for_tsv_file 
            df_for_evaluation_dict['gold'] += y_gold_for_tsv_file
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    filename_for_pred_file = "{0}_{1}.tsv".format(model_name.replace('.pt', ''), epoch_nr)
    df_for_evaluation = pd.DataFrame.from_dict(df_for_evaluation_dict)
    df_for_evaluation.to_csv(filename_for_pred_file, sep='\t', index=False)

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
