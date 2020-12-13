import joblib
import matplotlib as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', None)
data = pd.read_csv("cardio_train_clean_featureselection.csv")
print(data.shape)
print(data.head())

#let's look if data is balanced by looking at target variable
print(data['Cardio'].value_counts())
#there are 31181 values of 0 and 30922. Therefore our data is balanced.

sns.countplot(x='Cardio', data=data, palette='hls')
plt.show()
plt.savefig('Cariod_data_balance_check')
#checking percentage of each data variable
cardio_no = len(data[data['Cardio']==0])
cardio_yes = len(data[data['Cardio']==1])
percentage_of_no_CVD = cardio_no/(cardio_no+cardio_yes)
print("Percentage of people with no CVD is", percentage_of_no_CVD*100)
percentage_of_CVD = cardio_yes / (cardio_no + cardio_yes)
print("Percentage of people with CVD is", percentage_of_CVD*100)

#data is balanced so no need to do anymore balancing
#Percentage of people with no CVD is 50.20852454792845
#Percentage of people with CVD is 49.79147545207156

#now let us look at the relationship of the CVD to the features
print(data.groupby('Cardio').mean())

#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#set x as features and y as target variable
X = data[['Age', 'Gender','Height', 'Weight', "Systolic BP", "Chlosterol ", "Glucose", "Smoke", "Alcohol", "Active"]]
y = data['Cardio']
#train test 75/25 ratio
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
'''
#load logistic regression
logistic_regression_model= LogisticRegression(max_iter=100000, C=100)
logistic_regression_model.fit(X_train,y_train)
y_pred=logistic_regression_model.predict(X_test)
#confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()
print (X_test)
print (y_pred)

clf = LogisticRegression(max_iter = 10000).fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Model Evaluation metrics
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))

#Logistic Regression Classifier Confusion matrix - using test/train split hyperparameter tuning 
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))
#Grid Search


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
clf = LogisticRegression(max_iter = 10000)
grid_values = {'penalty': ['l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)
grid_clf_acc.fit(X_train, y_train)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(X_test)

# New Model Evaluation metrics
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))
print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))
print('F1 Score : ' + str(f1_score(y_test,y_pred_acc)))

#Logistic Regression (Grid Search) Confusion matrix
print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred_acc)))
'''
#using k-fold cross validation and tuning

model_lr = LogisticRegression(max_iter = 10000)
penalty = ['l2']
C = C = [0.001, 0.1, 1, 10, 100]
hyperparameters = dict(C=C, penalty = penalty)
from sklearn.model_selection import RandomizedSearchCV
param_tune = RandomizedSearchCV(model_lr, hyperparameters, random_state= 41)
param_tune.fit(X,y)
print("Best: %f using %s" % (param_tune.best_score_, param_tune.best_params_))
joblib.dump(param_tune,'logistic_regression_model.sav')
