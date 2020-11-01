import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)
data = pd.read_csv("cardio_train_clean.csv")
print(data.describe())

data_array = data.to_numpy()
n,m = data_array.shape

index = np.nonzero(data_array[:, -1] == 1)
k = len(index[0])
#print(index)
str = ['gender', 'chlosterol', 'glucose', 'smoke', 'alcohol', 'active']
id = [2, 7, 8, 9, 10, 11]
d = len(str)
bin_array = np.zeros(d)
for i in range(d):
    bin = 0
    for j in range(k):
        if data_array[index[0][j]][id[i]] == 1:
            bin += 1
    print("bin", bin)
    bin_array[i] = bin / k

print(bin_array)
bin_array2 = np.ones(d)
bin_array2 = bin_array2 - bin_array
print(bin_array2)
df = pd.DataFrame({'0': bin_array2, '1': bin_array}, index=str)
ax = df.plot.bar(title = 'If patient has CVD', rot=0)
plt.show()




'''# histogram for weight
fig = plt.figure(figsize=(10, 10))  # Define plot area
ax = fig.gca()  # Define axis
df =
plt.bar(data.loc[:, 'weight'])
ax.set_title('Box plot of weight (kg)')  # Give the plot a main title
ax.set_ylim(0.0, 300)
plt.show()'''