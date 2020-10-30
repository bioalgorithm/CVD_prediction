# data exploration
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)
data = pd.read_csv("cardio_train_updated.csv")

#print(data)
#print(data.dtypes)
#print(data.describe())

# Counts
"""data['counts'] = 1
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
"""
"""
PLOTS
"""

"""# boxplot for height
fig = plt.figure(figsize=(10, 10))  # Define plot area
ax = fig.gca()  # Define axis
plt.boxplot(data.loc[:, 'height'])
ax.set_title('Box plot of height (cm)')  # Give the plot a main title
ax.set_ylim(0.0, 300)
plt.show()

# boxplot for weight
fig = plt.figure(figsize=(10, 10))  # Define plot area
ax = fig.gca()  # Define axis
plt.boxplot(data.loc[:, 'weight'])
ax.set_title('Box plot of weight (kg)')  # Give the plot a main title
ax.set_ylim(0.0, 300)
plt.show()

# boxplot for app_hi
fig = plt.figure(figsize=(10, 10))  # Define plot area
ax = fig.gca()  # Define axis
plt.boxplot(data.loc[:, 'ap_hi'])
ax.set_title('Box plot of systolic blood pressure (mmHg)')  # Give the plot a main title
ax.set_ylim(0.0, 300)
plt.show()

fig = plt.figure(figsize=(10, 10))  # Define plot area
ax = fig.gca()  # Define axis
plt.boxplot(data.loc[:, 'ap_lo'])
ax.set_title('Box plot of diastolic blood pressure')  # Give the plot a main title
ax.set_ylim(0.0, 300)
plt.show()"""

"""
Counting Outliers
"""

# To Numpy
data_array = data.to_numpy()

# Height Checker
height = data_array[:, 3]
maxH = 180
minH = 140
maxOutlierH = height[height > maxH]
nOutMaxH = len(maxOutlierH)
minOutlierH = height[height < minH]
nOutMinH = len(minOutlierH)
totalHeight = nOutMaxH + nOutMinH

print("Height Outliers:")
print("Count > ", maxH, " is: ", nOutMaxH)
print("Count < ", minH, " is: ", nOutMinH)
print("Total Amount of Outliers (height): ", totalHeight)

# Weight Checker
weight = data_array[:, 4]
maxW = 110
minW = 40
maxOutlierW = weight[weight > maxW]
nOutMaxW = len(maxOutlierW)
minOutlierW = weight[weight < minW]
nOutMinW = len(minOutlierW)
totalWeight = nOutMaxW + nOutMinW

print("Weight Outliers:")
print("Count > ", maxW, " is: ", nOutMaxW)
print("Count < ", minW, " is: ", nOutMinW)
print("Total Amount of Outliers (weight): ", totalWeight)

# Systolic Blood Pressure
sys = data_array[:, 5]
maxS = 170
minS = 90
maxOutlierS = sys[sys > maxS]
nOutMaxS = len(maxOutlierS)
minOutlierS = sys[sys < minS]
nOutMinS = len(minOutlierS)
totalSys = nOutMaxS + nOutMinS

print("Systolic Outliers:")
print("Count > ", maxS, " is: ", nOutMaxS)
print("Count < ", minS, " is: ", nOutMinS)
print("Total Amount of Outliers (Systolic): ", totalSys)

# Diastolic Blood Pressure
dia = data_array[:, 6]
maxD = 110
minD = 70
maxOutlierD = dia[dia > maxD]
nOutMaxD = len(maxOutlierD)
minOutlierD = dia[dia < minD]
nOutMinD = len(minOutlierD)
totalDia = nOutMaxD + nOutMinD
print("Diastolic Outliers:")
print("Count > ", maxD, " is: ", nOutMaxD)
print("Count < ", minD, " is: ", nOutMinD)
print("Total Amount of Outliers (Diastolic): ", totalDia)

total = totalWeight + totalHeight + totalDia + totalSys

print("Total Outliers = ", total)
print("Good Data = ", 70000 - total)

heightGood = height[height <= maxH]
heightGood = heightGood[heightGood >= minH]

WGood = weight[weight <= maxW]
WGood = WGood[WGood >= minW]

SysGood = sys[sys <= maxS]
SysGood = SysGood[SysGood >= minS]

diaGood = dia[dia <= maxD]
diaGood = diaGood[diaGood >= minD]

full_data = np.copy(data_array)
# TO DO: Change gender, make male (2) into 0 value, for chlosterol/gluc change 1->0 2+3->1

# TO DO: We have the clean data in each col, but what we need to do is find the index of the outliers, we need to remove
#        the outliers row index, which would get rid of the entire row, rather than just the height, weight etc data.

# Pseudo: Find index, take initial data_array, use index to get rid of row in data_array, then take the final array
#         and make CSV from it, call this cardio_train_clean.csv

# TO DO: analyze newly cleaned data (cardio_train_clean)

