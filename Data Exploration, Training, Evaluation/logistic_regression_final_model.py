#import libraries
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', None)
data = pd.read_csv("cardio_train_clean_featureselection.csv")
print(data.shape)
print(data.head())

#logistic regression
from sklearn.linear_model import LogisticRegression
#set x as features and y as target variable, removing the alcohol and smoke columns
X = data[['Age', 'Gender','Height', 'Weight', "Systolic BP", "Chlosterol ", "Glucose", "Active"]]
y = data['Cardio']

#using k-fold cross validation and tuning
model_lr = LogisticRegression(max_iter = 10000)
penalty = ['l2']
C = [0.001, 0.1, 1, 10, 100]
hyperparameters = dict(C=C, penalty = penalty)
from sklearn.model_selection import RandomizedSearchCV
param_tune = RandomizedSearchCV(model_lr, hyperparameters, random_state= 41)
param_tune.fit(X,y)
print("Best: %f using %s" % (param_tune.best_score_, param_tune.best_params_))
joblib.dump(param_tune,'logistic_regression_final_model.sav')
