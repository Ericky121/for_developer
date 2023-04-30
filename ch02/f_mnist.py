import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

f