from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

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
y_encoded = to_categorical(y,num_classes=10)

# train test split data
X_train,X_test,y_train,y_test=train_test_split(X,y_encoded)


# neural net
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# # define layers of and compile model
# model = Sequential([
#     Dense(300, input_shape=(784,)),
#     Activation('sigmoid'),
#     Dense(300, input_shape=(300,)),
#     Activation('sigmoid'),
#     Dense(10),
#     Activation('softmax'),
# ])
#
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # fit model to train data
# model.fit(X_train, y_train, epochs=50, batch_size=400)

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(300, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)

model = MyModel()

print(model)
