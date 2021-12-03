from data_preprocessing import merge_data 

PathToTrainLabels = "../data/ClarificationTask_TrainLabels_Sep23.tsv"
PathToTrainData = "../data/ClarificationTask_TrainData_Sep23.tsv"

PathToDevLabels = "../data/ClarificationTask_DevLabels_Oct22a.tsv"
PathToDevData = "../data/ClarificationTask_DevData_Oct22a.tsv"

train_df = merge_data(PathToTrainData, PathToTrainLabels)
development_df = merge_data(PathToDevData, PathToDevLabels)

print(train_df)
print(development_df)