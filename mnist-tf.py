from __future__ import print_function
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# import, reindex, and validate data
mnist_train_small = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv", sep=",")

mnist_train_small = mnist_train_small.reindex(
    np.random.permutation(mnist_train_small.index))

print(mnist_train_small.describe())
