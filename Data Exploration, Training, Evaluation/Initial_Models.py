#import libraries
#includes all models except logistic regression
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
from sklearn.ensemble import RandomForestClassifier

#load dataset
data = pd.read_csv("cardio_train_clean_featureselection.csv")

# Features = xfeat, Target Variable = Y
xfeat = data[['Age', 'Gender', 'Height', 'Weight', "Systolic BP", "Chlosterol ", "Glucose", "Smoke", "Alcohol", "Active"]]
y = data['Cardio']

x_train, x_test, y_train, y_test = train_test_split(xfeat, y, test_size=0.2, random_state=0)

# LDA
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(x_train, y_train)

y_pred_LDA = lda_model.predict(x_test)
#confusion matrix
confusion_matrix_LDA = pd.crosstab(y_test, y_pred_LDA, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix_LDA, annot=True)
#print results
print()
print('LDA Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_LDA))
print('Precision: ', metrics.precision_score(y_test, y_pred_LDA))
print('Recall: ', metrics.recall_score(y_test, y_pred_LDA))
print()
#show plot
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

# Random Forests

rf_model = RandomForestClassifier(max_depth=5)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)

confusion_matrix_rf = pd.crosstab(y_test, y_pred_rf, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix_rf, annot=True)

print('Random Forest Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_rf))
print('Precision: ', metrics.precision_score(y_test, y_pred_rf))
print('Recall: ', metrics.recall_score(y_test, y_pred_rf))
print()
plt.show()

#model final matrix (the logistic regression model was modelled in a separate file)
"""

LDA Metrics:
Accuracy:  0.7149987923677643
Precision:  0.7452936746987951
Recall:  0.6441588024731533

QDA Metrics:
Accuracy:  0.6987360115932695
Precision:  0.7156440617151059
Recall:  0.6490400260331923

Gaussian Naive Bayes Metrics:
Accuracy:  0.6959182030432333
Precision:  0.7202082171407325
Recall:  0.6303286690530426

Multinomial Naive Bayes Metrics:
Accuracy:  0.6981724498832622
Precision:  0.7219033512312535
Recall:  0.6343963553530751

SVM Metrics:
Accuracy:  0.705901296191933
Precision:  0.755273397501536
Recall:  0.6000650829808005

Random Forest Metrics:
Accuracy:  0.712905563159166
Precision:  0.7303571428571428
Recall:  0.6654734786853238

Logistic Regression Metrics:
Accuracy: 0.7169264459616128
Precision: 0.7445868316394167
Recall: 0.6551321928460342

"""

