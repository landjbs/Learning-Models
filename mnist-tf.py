from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation

# import, reindex, and validate data
mnist_train_small = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv", sep=",")

mnist_train_small = mnist_train_small.reindex(
    np.random.permutation(mnist_train_small.index))

# function to process raw data
def preprocess_data(mnist_train_small):
  """
  Args: mnist_train_small dataframe of number shapes and labels
  Returns: dataframe of number image pixels (important features)
  and array of image labels (which number it is)
  """
  output_targets=pd.DataFrame()
  output_features=mnist_train_small.drop("6",axis=1)
  output_targets["label"]=mnist_train_small["6"]
  return output_features,output_targets

# preprocess data
X,y = preprocess_data(mnist_train_small)

# one-hot encode targets
y_encoded = to_categorical(y, num_classes=10)

# train test split data
X_train,X_test,y_train,y_test=train_test_split(X,y_encoded)

def make_model(X_train=X_train, y_train=y_train, save=True):
    k_model = Sequential()
    k_model.add(Dense(300, activation='sigmoid', input_shape=(784,)))
    k_model.add(Dense(300, activation='sigmoid'))
    k_model.add(Dense(10, activation='softmax'))
    k_model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # fit model to train data
    k_model.fit(X_train, y_train, epochs=50, batch_size=400)
    if save: pickle.dump(k_model, open('k_model.sav','wb'))
    return k_model

loaded_model = pickle.load(open("k_model.sav", "rb"))

print(loaded_model.predict(X_test))
