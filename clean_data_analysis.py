import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)
data = pd.read_csv("cardio_train_clean.csv")
print(data.describe())

data_array = data.to_numpy()
string = ["does not have CVD", "has CVD"]
n,m = data_array.shape
str = ['gender', 'chlosterol', 'glucose', 'smoke', 'alcohol', 'active']
id = [2, 7, 8, 9, 10, 11]
d = len(str)

for cvd in range(2):
    index = np.nonzero(data_array[:, -1] == cvd)
    k = len(index[0])
    #print(index)

    bin_array = np.zeros(d)

    for i in range(d):
        bin = 0
        for j in range(k):
            if data_array[index[0][j], id[i]] == 1:
                bin += 1
        bin_array[i] = bin / k

    bin_array2 = np.ones(d)
    bin_array2 -= bin_array
    df = pd.DataFrame({'0': bin_array2, '1': bin_array}, index=str)
    ax = df.plot.bar(title = 'If patient ' + string[cvd], rot=0)
    plt.ylim(0, 1)
    plt.show()

corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

corr_matrix = data.corr()
sns.heatmap(corr_matrix*corr_matrix, annot=True)
plt.show()


'''# histogram for weight
fig = plt.figure(figsize=(10, 10))  # Define plot area
ax = fig.gca()  # Define axis
df =
plt.bar(data.loc[:, 'weight'])
ax.set_title('Box plot of weight (kg)')  # Give the plot a main title
ax.set_ylim(0.0, 300)
plt.show()'''