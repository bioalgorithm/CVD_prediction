import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm


data = pd.read_csv("cardio_train_clean_featureselection.csv")

# Features = xfeat, Target Variable = Y
xfeat = data[['Age', 'Gender','Height', 'Weight', "Systolic BP", "Chlosterol ", "Glucose", "Smoke", "Alcohol", "Active"]]
y = data['Cardio']

x_train, x_test, y_train, y_test = train_test_split(xfeat, y, test_size=0.2, random_state=0)

# LDA
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(x_train, y_train)

y_pred_LDA = lda_model.predict(x_test)

confusion_matrix_LDA = pd.crosstab(y_test, y_pred_LDA, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix_LDA, annot=True)

print()
print('LDA Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_LDA))
print('Precision: ', metrics.precision_score(y_test, y_pred_LDA))
print('Recall: ', metrics.recall_score(y_test, y_pred_LDA))
print()

plt.show()

# QDA
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(x_train, y_train)

y_pred_QDA = qda_model.predict(x_test)

confusion_matrix_QDA = pd.crosstab(y_test, y_pred_QDA, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix_QDA, annot=True)
print('QDA Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_QDA))
print('Precision: ', metrics.precision_score(y_test, y_pred_QDA))
print('Recall: ', metrics.recall_score(y_test, y_pred_QDA))
print()
plt.show()

# Gaussian Naive Bayes

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred_gnb = gnb.predict(x_test)

confusion_matrix_gnb = pd.crosstab(y_test, y_pred_gnb, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix_gnb, annot=True)

print('Gaussian Naive Bayes Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_gnb))
print('Precision: ', metrics.precision_score(y_test, y_pred_gnb))
print('Recall: ', metrics.recall_score(y_test, y_pred_gnb))
print()
plt.show()

# Multinomial Naive Bayes

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred_mnb = mnb.predict(x_test)

confusion_matrix_mnb = pd.crosstab(y_test, y_pred_mnb, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix_mnb, annot=True)

print('Multinomial Naive Bayes Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_mnb))
print('Precision: ', metrics.precision_score(y_test, y_pred_mnb))
print('Recall: ', metrics.recall_score(y_test, y_pred_mnb))
print()
plt.show()

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
