from data_preprocessing import merge_data 
import torch 
from torchtext.legacy import data 
from transformers import BertTokenizer
import pdb 
from torch.nn import BCELoss, CrossEntropyLoss
from transformers import BertTokenizer, BertModel
from models import BERTClassification
import torch.optim as optim 
from helpers import train 

bert = BertModel.from_pretrained('bert-base-uncased')


# Dataset reading paths. 
PathToTrainLabels = "../data/ClarificationTask_TrainLabels_Sep23.tsv"
PathToTrainData = "../data/ClarificationTask_TrainData_Sep23.tsv"
PathToDevLabels = "../data/ClarificationTask_DevLabels_Oct22a.tsv"
PathToDevData = "../data/ClarificationTask_DevData_Oct22a.tsv"





# Model parameters 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
MAX_SEQ_LEN = 100


# set sequential = False, those fields are not texts. 
ids = data.Field()
label = data.Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
#label = data.LabelField(dtype=torch.float)
# set use_vocab=False to use own decoder 
version = data.Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
ids.build_vocab()
#label.build_vocab()
version.build_vocab()

fields = {'ids': ('ids', ids), 'version': ('version', version), 'label': ('label', label)}


def read_data(): 
    train_df = merge_data(PathToTrainData, PathToTrainLabels)
    development_df = merge_data(PathToDevData, PathToDevLabels)

    train_df.to_csv("./data/train.csv", index=False)
    development_df.to_csv("./data/dev.csv", index=False)


    train_data, valid_data, test_data = data.TabularDataset.splits(
                                            path = 'data',
                                            train = 'train.csv',
                                            validation = 'dev.csv',
                                            test = 'dev.csv', 
                                            format = 'csv',
                                            fields = fields,
                                            skip_header = False,
    )
    #label.build_vocab(train_data) 
    print("Train instances:", len(train_data)) 
    print("Dev instances:", len(valid_data))
    train_iter = data.BucketIterator(train_data, batch_size=16, sort_key=lambda x: len(x.version), train=True, sort=True, sort_within_batch=True)
    valid_iter = data.BucketIterator(valid_data, batch_size=16, sort_key=lambda x: len(x.version), train=True, sort=True, sort_within_batch=True)
    
    test_iter = data.Iterator(test_data, batch_size=16, train=False, shuffle=False, sort=False)
    return train_iter, valid_iter, test_iter


def main(): 

    # read data and return buckets 
    train_iter, valid_iter, test_iter = read_data()
    
    # create instance of the model 
    HIDDEN_DIM = 256
    OUTPUT_DIM = 3
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    

    model = BERTClassification(bert,
                            HIDDEN_DIM,
                            OUTPUT_DIM,
                            N_LAYERS,
                            BIDIRECTIONAL,
                            DROPOUT)
    

    optimizer = optim.Adam(model.parameters())

    criterion = CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)
    
    N_EPOCHS = 5
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iter, optimizer, criterion, device)
        print(train_loss, train_acc)
      
main()