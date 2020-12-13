#includes all models
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
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

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

#logistic regression
'''
#set x as features and y as target variable
X = data[['Age', 'Gender','Height', 'Weight', "Systolic BP", "Chlosterol ", "Glucose", "Smoke", "Alcohol", "Active"]]
y = data['Cardio']
#train test 75/25 ratio
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
'''

#load logistic regression
logistic_regression_model= LogisticRegression(max_iter=100000, C=100)
logistic_regression_model.fit(x_train,y_train)
y_pred_lr=logistic_regression_model.predict(x_test)

#confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred_lr, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)

print('Logistic Regression Metrics:')
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred_lr))
print('Precision: ', metrics.precision_score(y_test, y_pred_lr))
print('Recall: ', metrics.recall_score(y_test, y_pred_lr))

plt.show()
'''
print(x_test)
print(y_pred_lr)
'''

'''
Initial Untuned Metrics

LDA Metrics:
Accuracy:  0.7177360921020852
Precision:  0.7509505703422054
Recall:  0.6426944354051416

QDA Metrics:
Accuracy:  0.6799774575315997
Precision:  0.7310065971483294
Recall:  0.5589000976244712

Gaussian Naive Bayes Metrics:
Accuracy:  0.6561468480798648
Precision:  0.6990023349607302
Recall:  0.5357956394402864

Multinomial Naive Bayes Metrics:
Accuracy:  0.7026809435633202
Precision:  0.7333967649857279
Recall:  0.6270745200130166

SVM Metrics:
Accuracy:  0.7059818050076483
Precision:  0.7563733552631579
Recall:  0.5986007159127888

Random Forest Metrics:
Accuracy:  0.718702197890669
Precision:  0.738403451995685
Recall:  0.668239505369346

Logistic Regression Metrics:
Accuracy:  0.7190242331535303
Precision:  0.7459259259259259
Recall:  0.6553856166612431
'''