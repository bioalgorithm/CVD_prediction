# data exploration
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)
data = pd.read_csv("cardio_train_updated.csv")
data_array = data.to_numpy()


# Get rid of outliers
max = [180, 110, 170, 110]
min = [140, 40, 90, 70]
id = [3, 4, 5, 6]

for i in range(len(max)):
    index = np.nonzero(data_array[:, id[i]] > max[i])
    data_array = np.delete(data_array, index[0], axis=0)

    index = np.nonzero(data_array[:, id[i]] < min[i])
    data_array = np.delete(data_array, index[0], axis=0)

data_array = np.insert(data_array, 8, np.zeros(len(data_array[:,8])), axis=1)
data_array = np.insert(data_array, 9, np.zeros(len(data_array[:,8])), axis=1)

data_array = np.insert(data_array, 11, np.zeros(len(data_array[:,8])), axis=1)
data_array = np.insert(data_array, 12, np.zeros(len(data_array[:,8])), axis=1)

#data cleaning
# male (2->0)
index = np.nonzero(data_array[:, 2] == 2)
for i in index:
    data_array[i, 2] = 0
'''
# chlo (1->0) 7
index = np.nonzero(data_array[:, 7] == 1)
for i in index:
    data_array[i, 7] = 1
'''

# chlo if 2, put 1 hot column 8 to 1, and others to 0
index = np.nonzero(data_array[:, 7] == 2)
for i in index:
    data_array[i, 8] = 1
    data_array[i, 7] = 0

# chlo if 3, put 1 hot column 9 to 1, and others to 0
index = np.nonzero(data_array[:, 7] == 3)
for i in index:
    data_array[i, 9] = 1
    data_array[i, 7] = 0

'''
# gluc (1->0) 10
index = np.nonzero(data_array[:, 10] == 1)
for i in index:
    data_array[i, 8] = 0
'''

# gluc same as chlo from above
index = np.nonzero(data_array[:, 10] == 2)
for i in index:
    data_array[i, 11] = 1
    data_array[i, 10] = 0

index = np.nonzero(data_array[:, 10] == 3)
for i in index:
    data_array[i, 12] = 1
    data_array[i, 10] = 0

np.savetxt("cardio_train_clean_1hot.csv", data_array, fmt = '%i', delimiter=",", header= "id")

data_clean = pd.read_csv("cardio_train_clean_1hot.csv")
print(data.describe())
print(data_clean.describe())