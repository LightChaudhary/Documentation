## California Housing Prices

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("housing.csv")

# Checking for the first and last five rows of the datasets to get possible insight

print(df.head())
print(df.tail())

# Looking for the information about the d types and missing values percolumn. 

print(df.info())

# Description about the datasets 

print(df.describe())

# Visualize it 

df.hist(bins= 50, figsize=(12,10))
plt.show()

# Checking the number of houses according to their area/ocean proximity

print(df["ocean_proximity"].value_counts())

# Visualizing the Houses according to the ocean proximity 

df["ocean_proximity"].hist(bins = 50, figsize=(5,2))
plt.show()

# Creating a train and test set

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size= 0.2, random_state= 42)

# Stratified Sampling 

df["income_category"] = pd.cut(df["median_income"], bins= [0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
df["income_category"].hist(bins = 50, figsize=(10,8))
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state= 42)

for train_idx, test_idx in split.split(df, df["income_category"]) : 
    strat_train_set = df.loc[train_idx]
    strat_test_set = df.loc[test_idx]

for set_ in (strat_train_set, strat_test_set): 
    set_.drop("income_category", axis = 1, inplace = True)