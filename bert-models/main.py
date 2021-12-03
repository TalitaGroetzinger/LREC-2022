from data_preprocessing import merge_data 
from torchtext.legacy import data 

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