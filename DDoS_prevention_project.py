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
def plotMissingValues(dataframe):
    missing_values=dataframe.isnull().sum() #Counting null values for each
    fig=plt.figure(figuresize=(16,5))
    missing_values.plot(kind='bar')
    plt.xlabel("Features")
    plt.ylabel("Missing Values")
    plt.title("Total number of Missing values in each feature")
    plt.show()

plotMissingValues(df)
## Removing the null values
data_f=df.dropna()
#Checking the null values in the dataset
plt.figure(1,figsize=(10,4))
plt.hist(data_f.isna().sum())
#Set the title and axis Labels
plt.title('Data after remnoving the NUll Values')
plt.xlabel( 'The number of null values')
plt.ylabel('Number of columns')

#Show the plot
plt.show()
pd.set_option('use_inf_as _na',True) 
null_values=data_f.isnull().sum()
(data_f.dtypes=='object')

data_f['Label']=data_f['Label'].map({'BENINGN':0,'DDoS':1})

plt.hist(data_f['Label'],bins=[0,0.3,0.7,1],edgecolor='black')
plt.xticks([0,1],labels=['BENIGN','DDoS=1'])
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()

df.describe()

plt.figure(5)
for col in data_f.columns:
    plt.hist(data_f[col])
    plt.title(col)
    plt.show()
#Convert into numpy array
# 
# X1=np.array(data_f).astype(np.float64)
# y1=np.array(data-f['Label'])
#
# Split data into features and target variable
X=data_f.drop('Label',axis=1)
y=data_f['Label']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0,30,random_state=42)
print("The train dataset size=",X_train.shape)
print("The test dataset size=",X_test.shape)
#Training the model
#Random Forest
rf_model=RandomForestClassifier(n_estimators=50,random_state=42)
rf_model.fit(X_train,y_train)
rf_pred=rf_model.predict(X_test)
#Getting deature importances from the trained model
importances=rf_model.feature_importances_
#Getting the indices of features sorted by importance
indices=sorted(range(len(importances)),key=lambda i: importances[i],reverse=)
feature_names=[f"Features{i}"for i in indices] #replace with your coloumn

plt.figure(figsize=(8,14))
plt.barh(range(X_train.shape[1]),importances[indices],align="center")
plt.yticks(range(X_train.shape[1]),feature_names)
plt.xlabel("Importance")
plt.title("Feature Imprtances")
plt.show()
from sklearn.tree import plot_tree

estimator=rf_model.estimator_[0] #Selecting the first estimator from the 

plt.figure(figsize=(20,10))
plot_tree(estimator,filled=True,rounded=True)
plt.show()
#Model Evaluation 
def plot_confusion_matrix(y_true,y_pred)
    cm=plot_confusion_matrix(y_true=True,fmt='d',cmap='Blues',xticklabels=classes,)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
#Evaluate Random Forest
rf_accuracy=accuracy_score(y_test,rf_pred)
rf_f1=f1_score(y_test,rf_pred)
rf_precision=precision_score(y_test,rf_pred)
rf_recall=recall_score(y_test,rf_pred)

print('\nRandom Forest Matrices:')
print(f'Accuracy:{rf_accuracy:.4f}')
print(f'F1 Score:{rf_f1:.4f}')
print(f'Precision:{rf_precision:.4f}')
print(f'Recall:{rf_recall:.4f}')
#Confusion Matrix for Random Forest
plot_confusion_matrix(y_test,rf_pred,['Benign','DDoS'],'Random Forest Confusion Matrix')
#Logistic Regression
lr_model=LogisticRegression(random_state=42)
lr_model.fit(X_train,y_train)
lr_pred=(lr_model.predict(X_test))
#Evaluate Logistic Regression
lr_accuracy=accuracy_score(y_test,lr_pred)
lr_f1=f1_score(y_test,lr_pred)
lr_precision=precision_score(y_test,lr_pred)
lr_recall=recall_score(y_test,lr_pred)
print('\nLogistic Regression Metrics:')
print(f'Acurracy:{lr_accuracy:.4}')
print(f'F1 Score:{lr_f1:.4f}')
print(f'Precision:{lr_precision:.4f}')
print(f'Recall:{lr_recall:.4f}')
#
plot_confusion_matrix(y_test,lr_pred,['Benign','DDoS'],'Logistic Regression Confusion Matrix')
#Neural Network
nn_model=MLPClassifier(hidden_layer_sizes=(10,),max_iter=10,random_state=42)
nn_model.fit(X_train,y_train)
nn_pred=nn_model.predict(X_test)
#Evaluate Neural Network
nn_accuracy=accuracy_score(y_test,nn_pred)
nn_f1=f1_score(y_test,nn_pred)
nn_precision=precision_score(y_test,nn_pred)
nn_recall=recall_score(y_test,nn_pred)
print('\nLogistic Regression Metrics:')
print(f'Acurracy:{nn_accuracy:.4}')
print(f'F1 Score:{nn_f1:.4f}')
print(f'Precision:{nn_precision}')
print(f'Recall:{nn_recall:.4f}')
#Confusion Matrix For Neural Network
plot_confusion_matrix(y_test,nn_pred,['benign','DDoS'],'Neural Network Confusion Matrix')
#Model Comparision
#Random Forest\
rf_proba=rf_model.predict_proba(X_test)
lr_proba=lr_model.predict_proba(X_test)
nn_proba=nn_model.predict_proba(X_test)
# Combine prediction for ROC curve
rf_fpr,rf_tpr,_=roc_curve(y_test,rf_proba[:,1])
rf_auc=auc(rf_fpr,rf_tpr)
lr_fpr,lr_tpr,_=roc_curve(y_test,lr_proba[:,1])
lr_auc=auc(lr_fpr,lr_tpr)
nn_fpr,nn_tpr,_=roc_curve(y_test,nn_proba[:,1])
nn_auc=auc(nn_fpr,nn_tpr)
#Plot Roc vurves for all model 
plt.figure(figsize=(8,6))
plt.plot(rf_fpr,rf_tpr,label=f'Random Forest (AUC={rf_auc:.2f})')
plt.plot(lr_fpr,lr_tpr,label=f'Logistic Regression (AUC={lr_auc:.2f})')
plt.plot(nn_fpr,nn_tpr,label=f'Neural Network (AUC={nn_auc:.2f})')
#PLot ROc curve for random calssifier (50% area)
plt.plot([0,1],[0,1],linestyle='--',color='black',label='Random Classifier (AUC=0.50)')
plt.xlabel([0,1],[0,1],linestyle=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       )
