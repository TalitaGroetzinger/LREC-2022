import pandas as pd
from pandas.io.stata import precision_loss_doc 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv("predictions.tsv", sep='\t')
print(df)

preds = df['preds'].tolist()
gold = df['gold'].tolist()
gold_without_neutral = [pred for pred in gold if pred !=1]
label_distribution = Counter(gold)
print(label_distribution)

counter = 0 
for pred, gold_pred in zip(preds, gold): 
       if pred == gold_pred: 
           counter +=1 

accuracy = counter/len(preds)
print("normal accuracy", accuracy)


counter = 0 
for pred, gold_pred in zip(preds, gold): 
       if pred == gold_pred and gold_pred != 1: 
          counter +=1 


accuracy = counter/len(gold_without_neutral)
print("accuracy_without_neutral", accuracy)


cm = confusion_matrix(df['gold'], df['preds'])
cm = ConfusionMatrixDisplay(cm)
cm.plot()
plt.show()