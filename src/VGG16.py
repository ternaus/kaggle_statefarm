from __future__ import division
'''
This script trains keras NN based on VGG16 and saves model to file
'''

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

    for layer in model.layers:
        layer.trainable = False


    model.add(Dense(10, activation='softmax'))

    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # Learning rate is changed to 0.001

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
    	loss='categorical_crossentropy',
    	metrics=['accuracy'])
    return model

def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + cross + '.json'
    weight_name = 'model_weights' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)

if __name__ == "__main__":
  np.random.seed(random_state)

  data_root = os.path.join(os.path.expanduser('~'), 'workspace', 'kaggle_statefarm', 'data')

  print 'reading train'
  f = h5py.File(os.path.join(data_root, 'train_224_224.h5'),'r')

  X_train = np.array(f['X_train'], dtype=np.float32)
  y_train = np.array(f['y_train'], dtype=np.uint8)
  f.close()

  y_train = np_utils.to_categorical(y_train, 10)
  print X_train.shape, y_train.shape

  batch_size = 32
  random_state = 2016
  nb_epoch = 20
  split = 0.2
  img_rows, img_cols = 224, 224

  print 'creating model'
  model = vgg_std16_model(img_rows, img_cols)



	print 'fitting model'
	model.fit(X_train, y_train, batch_size=batch_size,
	        nb_epoch=nb_epoch,
	        verbose=1,
	        validation_split=split,
	        shuffle=True)

	print 'saving model'
	save_model(model, 'VGG_{batch_size}_{nb_epoch}_'.format(batch_size=batch_size, nb_epoch=nb_epoch))