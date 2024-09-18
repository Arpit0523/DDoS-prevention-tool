import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Machine learning Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

df pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df.head(3)

#data processing
df.columns = df.columns.str.strip()
#Unique values in the labbel target column
df.locx[:,'Lablel'].unique()

#checking fot null value in the dataset.
plt.figure(1,figssize=(10,4))
plt.hist(df.isna().sum())
#SET the title and axis labels
plt.xticks([0,1],labels=[Not Null=0,Null=1])
plt.title('Columns with NUull Values')
plt.xlabel('feature')
plt.ylabel('the number of features')

#Show the plot
plt.show()
