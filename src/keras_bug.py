import numpy as np

import os
import glob
import cv2
import math
import h5py
import datetime
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D


from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json
from numpy.random import permutation
import time

random_state = 2016


def vgg_std16_model(img_rows, img_cols):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3,
                                                 img_rows, img_cols)))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    model.load_weights('../../models/vgg16_weights.h5')

    model.layers.pop()
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
      loss='categorical_crossentropy',
      metrics=['accuracy'])
    return model

data_root = os.path.join(os.path.expanduser('~'), 'workspace', 'kaggle_statefarm', 'data')

f = h5py.File(os.path.join(data_root, 'train_224_224.h5'),'r')

X_train = np.array(f['X_train'][:1], dtype=np.float32)
y_train = np.array(f['y_train'][:1], dtype=np.uint8)
f.close()

model = vgg_std16_model(224, 224)

print model.predict(X_train)

json_string = model.to_json()
open('my_model_architecture.json', 'w').write(json_string)
model.save_weights('my_model_weights.h5')


model_new = model_from_json(open('my_model_architecture.json').read())
model_new.load_weights('my_model_weights.h5')

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model_new.compile(optimizer=sgd,
  loss='categorical_crossentropy',
  metrics=['accuracy'])

print
print model_new.predict(X_train)
