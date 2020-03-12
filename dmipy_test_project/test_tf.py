import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.datasets import mnist
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import utils

print('Succesful Imports')

#with tf.device('XLA_GPU:1'):
#    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#    c = tf.matmul(a, b)

#with tf.Session() as sess:
#    print (sess.run(c))
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = utils.to_categorical(y_train, 10)
Y_test = utils.to_categorical(y_test, 10)

#le = LabelEncoder()
#le.fit(y_train)
#Y_train = le.transform(y_train)
#Y_test = le.transform(y_test)

model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=10000, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print("Final Test Score is {}".format(score))

#with tf.Session() as sess:
#  devices = sess.list_devices()

#print(devices)
