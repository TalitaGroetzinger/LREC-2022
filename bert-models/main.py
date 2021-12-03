from data_preprocessing import merge_data 
from torchtext.legacy import data 
from transformers import BertTokenizer, BertForSequenceClassification


PathToTrainLabels = "../data/ClarificationTask_TrainLabels_Sep23.tsv"
PathToTrainData = "../data/ClarificationTask_TrainData_Sep23.tsv"

PathToDevLabels = "../data/ClarificationTask_DevLabels_Oct22a.tsv"
PathToDevData = "../data/ClarificationTask_DevData_Oct22a.tsv"

train_df = merge_data(PathToTrainData, PathToTrainLabels)
development_df = merge_data(PathToDevData, PathToDevLabels)

train_df.to_csv("./data/train.csv", index=False)
development_df.to_csv("./data/dev.csv", index=False)

ids = data.Field()
version = data.Field()
label = data.Field()

#     merged_df_dict = {"ids": [], "version": [], "label": []}
fields = {'ids': ('ids', ids), 'version': ('version', version), 'label': ('label', label)}


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model parameter
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)


train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = 'data',
                                        train = 'train.csv',
                                        validation = 'dev.csv',
                                        test = 'dev.csv', 
                                        format = 'csv',
                                        fields = fields,
                                        skip_header = False,
)

print(vars(train_data[0]))


train_iter = data.BucketIterator(train_data, batch_size=16, sort_key=lambda x: len(version.text), train=True, sort=True, sort_within_batch=True)
valid_iter = data.BucketIterator(valid_data, batch_size=16, sort_key=lambda x: len(version.text), train=True, sort=True, sort_within_batch=True)
test_iter = data.Iterator(test_data, batch_size=16, train=False, shuffle=False, sort=False)

