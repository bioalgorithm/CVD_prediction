import numpy as np
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data = pd.read_csv("cardio_train_clean_featureselection.csv")

# Features = xfeat, Target Variable = Y
xfeat = data[['Age', 'Gender','Height', 'Weight', "Systolic BP", "Chlosterol ", "Glucose", "Smoke", "Alcohol", "Active"]]
y = data['Cardio']

x_train, x_test, y_train, y_test = train_test_split(xfeat, y, test_size=0.2, random_state=0)

# LDA
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(x_train, y_train)

y_pred = lda_model.predict(x_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)

print()
print('LDA Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
print('Precision: ', metrics.precision_score(y_test, y_pred))
print('Recall: ', metrics.recall_score(y_test, y_pred))
print()

plt.show()

# QDA
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(x_train, y_train)

y_pred = qda_model.predict(x_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
print('QDA Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
print('Precision: ', metrics.precision_score(y_test, y_pred))
print('Recall: ', metrics.recall_score(y_test, y_pred))

plt.show()