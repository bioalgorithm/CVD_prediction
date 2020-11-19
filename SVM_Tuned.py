import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn import svm

data = pd.read_csv("cardio_train_clean_featureselection.csv")

# Features = xfeat, Target Variable = Y
xfeat = data[['Age', 'Gender','Height', 'Weight', "Systolic BP", "Chlosterol ", "Glucose", "Smoke", "Alcohol", "Active"]]
y = data['Cardio']

x_train, x_test, y_train, y_test = train_test_split(xfeat, y, test_size=0.2, random_state=0)


# SVM

svm_model = svm.SVC()
svm_model.fit(x_train, y_train)
y_pred_svm = svm_model.predict(x_test)

confusion_matrix_svm = pd.crosstab(y_test, y_pred_svm, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix_svm, annot=True)

print('SVM Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_svm))
print('Precision: ', metrics.precision_score(y_test, y_pred_svm))
print('Recall: ', metrics.recall_score(y_test, y_pred_svm))
print()
plt.show()