#import relevant libraries
#ensure plots are shown in jupyter notebook
#%matplotlib

import os
import time
import pickle
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get the data

URL_Red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
def fetch_data_red(fname = "winequality-red.csv"):
    response = requests.get(URL_Red)
    outpath = os.path.abspath(fname)
    with open(outpath, "wb") as f:
        f.write(response.content)
    return outpath

URL_White = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
def fetch_data_white(fname = "winequality-white.csv"):
    response = requests.get(URL_White)
    outpath = os.path.abspath(fname)
    with open(outpath, "wb") as f:
        f.write(response.content)
    return outpath

# bring in the data

WhiteData = fetch_data_white()
df_white = pd.read_csv(WhiteData, sep = ";", header = 0)

RedData = fetch_data_red()
df_red = pd.read_csv(RedData, sep = ";", header = 0)

# save features and labels

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

LABEL_MAP = [
    "quality"
]

print(df_red.describe())


# Determine the shape of the data
print("{} instances with {} features\n".format(*df_red.shape))

# Determine the frequency of each class
print(df_red.groupby('quality').count())


%matplotlib inline

# Create a scatter matrix of the dataframe features
from pandas.tools.plotting import scatter_matrix
scatter_matrix(df_red, alpha=0.2, figsize=(12, 12), diagonal='kde')
plt.show()


from pandas.tools.plotting import parallel_coordinates
plt.figure(figsize=(12,12))
parallel_coordinates(df_red, 'quality')
plt.show()


from pandas.tools.plotting import radviz
plt.figure(figsize=(12,12))
radviz(df_red, 'quality')
plt.show()

#next start the scikit learn portion. regression or
#good for this data 
