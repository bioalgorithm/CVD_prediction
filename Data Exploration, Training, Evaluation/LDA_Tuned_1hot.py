import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import joblib

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


data = pd.read_csv("data/cardio_train_clean_1hot_featureselection.csv")

# Features = xfeat, Target Variable = Y
xfeat = data[['Age', 'Gender','Height', 'Weight', "Systolic BP",
              "Chlosterol_Normal","Chlosterol_High","Chlosterol_Veryhigh",
              "Glucose_Normal","Glucose_High","Glucose_Veryhigh", "Smoke", "Alcohol", "Active"]]
y = data['Cardio']

x_train, x_test, y_train, y_test = train_test_split(xfeat, y, test_size=0.2, random_state=0)

# LDA
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(x_train, y_train)

y_pred_LDA = lda_model.predict(x_test)
print(x_test)
y_prob = lda_model.predict_proba(x_test)
confusion_matrix_LDA = pd.crosstab(y_test, y_pred_LDA, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix_LDA, annot=True)

print()
print('INITIAL LDA Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_LDA))
print('Precision: ', metrics.precision_score(y_test, y_pred_LDA))
print('Recall: ', metrics.recall_score(y_test, y_pred_LDA))
print()


# Tuned LDA
lda_model = LinearDiscriminantAnalysis()

parameters = {'solver': ['lsqr', 'eigen']}

search = RandomizedSearchCV(lda_model, parameters, scoring='recall')
results = search.fit(xfeat, y)

joblib.dump(search, 'Lda_Tuned.sav')

print('Mean accuracy: ', results.best_score_)
print('Config: ', results.best_params_)

confusion_matrix_LDA = pd.crosstab(y_test, y_pred_LDA, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix_LDA, annot=True)

print()
print('Tuned LDA Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_LDA))
print('Precision: ', metrics.precision_score(y_test, y_pred_LDA))
print('Recall: ', metrics.recall_score(y_test, y_pred_LDA))
print()


