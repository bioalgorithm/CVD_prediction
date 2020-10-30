#data exploration
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
data = pd.read_csv("cardio_train_updated.csv")

print(data)

print(data.dtypes)

print(data.describe())

data['counts'] = 1
print(data[['counts', 'gluc']].groupby(['gluc']).agg('count'))
data['counts'] = 1
print(data[['counts', 'weight']].groupby(['weight']).agg('count'))
data['counts'] = 1
print(data[['counts', 'height']].groupby(['height']).agg('count'))
data['counts'] = 1
print(data[['counts', 'smoke']].groupby(['smoke']).agg('count'))
data['counts'] = 1
print(data[['counts', 'alco']].groupby(['alco']).agg('count'))
data['counts'] = 1
print(data[['counts', 'active']].groupby(['active']).agg('count'))
data['counts'] = 1
print(data[['counts', 'cardio']].groupby(['cardio']).agg('count'))
data['counts'] = 1
print(data[['counts', 'age']].groupby(['age']).agg('count'))
data['counts'] = 1
print(data[['counts', 'ap_hi']].groupby(['ap_hi']).agg('count'))
data['counts'] = 1
print(data[['counts', 'cholesterol']].groupby(['cholesterol']).agg('count'))

#boxplot for height
fig = plt.figure(figsize=(10, 10)) # Define plot area
ax = fig.gca() # Define axis
plt.boxplot(data.loc[:, 'height'])
ax.set_title('Box plot of height (cm)') # Give the plot a main title
ax.set_ylim(0.0, 300)
plt.show()


#boxplot for weight
fig = plt.figure(figsize=(10, 10)) # Define plot area
ax = fig.gca() # Define axis
plt.boxplot(data.loc[:, 'weight'])
ax.set_title('Box plot of weight (kg)') # Give the plot a main title
ax.set_ylim(0.0, 300)
plt.show()

#boxplot for app_hi
fig = plt.figure(figsize=(10, 10)) # Define plot area
ax = fig.gca() # Define axis
plt.boxplot(data.loc[:, 'ap_hi'])
ax.set_title('Box plot of systolic blood pressure (mmHg)') # Give the plot a main title
ax.set_ylim(0.0, 300)
plt.show()


fig = plt.figure(figsize=(10, 10)) # Define plot area
ax = fig.gca() # Define axis
plt.boxplot(data.loc[:, 'ap_lo'])
ax.set_title('Box plot of diastolic blood pressure') # Give the plot a main title
ax.set_ylim(0.0, 300)
plt.show()


data_array = data.to_numpy()

#Height Checker
height = data_array[:, 3]
maxH = 180
minH = 140
maxOutlier = height[height>maxH]
nOutMax = len(maxOutlier)
print("Count > ", maxH, " is: ", nOutMax)

minOutlier = height[height<minH]
nOutMin = len(minOutlier)
print("Count < ", minH, " is: ", nOutMin)

print("Total Amount of Outliers: ", nOutMax + nOutMin)

#Weight Checker
height = data_array[:, 4]
maxH = 180
minH = 140
maxOutlier = height[height>maxH]
nOutMax = len(maxOutlier)
print("Count > ", maxH, " is: ", nOutMax)

minOutlier = height[height<minH]
nOutMin = len(minOutlier)
print("Count < ", minH, " is: ", nOutMin)

print("Total Amount of Outliers: ", nOutMax + nOutMin)

#print(data_array)
