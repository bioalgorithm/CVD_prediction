#data exploration
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("cardio_train_updated.csv")
print(data)

print(data.dtypes)

data.describe()

data['counts'] = 1
print(data[['counts', 'gluc']].groupby(['gluc']).agg('count'))
data['counts'] = 1
print(data[['counts', '']].groupby(['gluc']).agg('count'))







