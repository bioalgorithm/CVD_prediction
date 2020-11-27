import numpy as np
from sklearn import metrics
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("cardio_train_clean_featureselection.csv")
data_1hot = pd.read_csv("cardio_train_clean_1hot_featureselection.csv")
# Features = xfeat, Target Variable = Y
xfeat = data[['Age', 'Gender','Height', 'Weight', "Systolic BP",
              "Chlosterol ", "Glucose", "Smoke", "Alcohol", "Active"]]

xfeat_1hot = data_1hot[['Age', 'Gender','Height', 'Weight', "Systolic BP",
              "Chlosterol_Normal","Chlosterol_High","Chlosterol_Veryhigh",
              "Glucose_Normal","Glucose_High","Glucose_Veryhigh", "Smoke", "Alcohol", "Active"]]

y = data['Cardio']

x_train, x_test, y_train, y_test = train_test_split(xfeat, y, test_size=0.2, random_state=0)
x_train_1hot, x_test_1hot, y_train_1hot, y_test_1hot = train_test_split(xfeat_1hot, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier()

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune



# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator=rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(xfeat_1hot, y)

best_random = rf_random.best_estimator_
joblib.dump(rf_random, 'random_forest.sav')
print('Config: ', rf_random.best_params_)


'''
rf.fit(x_train_1hot, y_train_1hot)
y_pred_rf = rf.predict(x_test_1hot)

print()
print('INITIAL_1hot rf Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test_1hot, y_pred_rf))
print('Precision: ', metrics.precision_score(y_test_1hot, y_pred_rf))
print('Recall: ', metrics.recall_score(y_test_1hot, y_pred_rf))
print()

rf = RandomForestClassifier(n_estimators= 1000, min_samples_split = 10, min_samples_leaf = 2, max_features = 'sqrt',
                            max_depth= 10, bootstrap = True)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print()
print('Tuned_1hot rf Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test_1hot, y_pred_rf))
print('Precision: ', metrics.precision_score(y_test_1hot, y_pred_rf))
print('Recall: ', metrics.recall_score(y_test_1hot, y_pred_rf))
print()

'''

'''

rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print()
print('INITIAL rf Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_rf))
print('Precision: ', metrics.precision_score(y_test, y_pred_rf))
print('Recall: ', metrics.recall_score(y_test, y_pred_rf))
print()

rf = RandomForestClassifier(n_estimators= 200, min_samples_split = 5, min_samples_leaf = 4, max_features = 'auto',
                            max_depth= 10, bootstrap = True)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print()
print('Tuned rf Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_rf))
print('Precision: ', metrics.precision_score(y_test, y_pred_rf))
print('Recall: ', metrics.recall_score(y_test, y_pred_rf))
print()

'''

'''
Config:  {'n_estimators': 1000, 'min_samples_split': 10, 
'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True}
'''