import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('cardio_train_clean_featureselection.csv')
data_1hot = pd.read_csv('cardio_train_clean_1hot_featureselection.csv')

rf = joblib.load('random_forest.sav')
lr = joblib.load('logistic_regression_model.sav')

X = data[['Age', 'Gender','Height', 'Weight', "Systolic BP", "Chlosterol ", "Glucose", "Smoke", "Alcohol", "Active"]]
y = data['Cardio']

X_1 = data_1hot[['Age', 'Gender','Height', 'Weight', "Systolic BP",
              "Chlosterol_Normal","Chlosterol_High","Chlosterol_Veryhigh",
              "Glucose_Normal","Glucose_High","Glucose_Veryhigh", "Smoke", "Alcohol", "Active"]]
y_1 = data_1hot['Cardio']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
x_train2, x_test2, y_train2, y_test2 = train_test_split(X_1, y_1, test_size=0.25, random_state=0)

y_pred_lr = lr.predict(x_test)
y_pred_rf = rf.predict(x_test2)

print('Logistic Regression Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_lr))
print('Precision: ', metrics.precision_score(y_test, y_pred_lr))
print('Recall: ', metrics.recall_score(y_test, y_pred_lr))

print()
print('Random Forest Metrics:')
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_rf))
print('Precision: ', metrics.precision_score(y_test, y_pred_rf))
print('Recall: ', metrics.recall_score(y_test, y_pred_rf))
print()

'''
Logistic Regression Metrics:
Accuracy:  0.717184078320237
Precision:  0.7445212531254596
Recall:  0.6560393986521513

Random Forest Metrics:
Accuracy:  0.7341878139894371
Precision:  0.7626994583516322
Recall:  0.6752203214100571
'''