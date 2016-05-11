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
import cPickle as pickle
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

    model.layers.pop() # Get rid of the classification layer
    # model.layers.pop() # Get rid of the dropout layer
    # model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []

    for layer in model.layers[:-1]:
        layer.trainable = False

    # model.add(Dropout(0.5))

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

  img_rows, img_cols = 224, 224
  batch_size = 32
  nb_epoch = 20
  num_images = 79726
  n_folds = 2
  np.random.seed(random_state)

  data_root = os.path.join(os.path.expanduser('~'), 'workspace', 'kaggle_statefarm', 'data')
  cache_path = os.path.join(os.path.expanduser('~'), 'workspace', 'kaggle_statefarm', 'src', 'cache')

  drivers = pd.read_csv(os.path.join(data_root, 'driver_imgs_list.csv'))

  unique_drivers = drivers['subject'].unique()

  kf = KFold(len(unique_drivers), n_folds=n_folds,
               shuffle=True, random_state=random_state)

  print 'reading train'
  f = h5py.File(os.path.join(data_root, 'train_224_224.h5'),'r')

  now = datetime.datetime.now()
  suffix = str(now.strftime("%Y-%m-%d-%H-%M"))

  ind = 0
  for train_drivers, test_drivers in kf:
    y_train = []
    X_train = []
    y_val = []
    X_val = []

    for driver in unique_drivers[train_drivers]:
      X_train += [np.array(f['X_{driver}'.format(driver=driver)])]
      y_train += [np.array(f['y_{driver}'.format(driver=driver)])]

    for driver in unique_drivers[test_drivers]:
      X_val += [np.array(f['X_{driver}'.format(driver=driver)])]
      y_val += [np.array(f['y_{driver}'.format(driver=driver)])]

    print 'shuffling'

    X_train = np.vstack(X_train).astype(np.float32)
    y_train = np.hstack(y_train)

    X_val = np.vstack(X_val).astype(np.float32)
    y_val = np.hstack(y_val)

    mean_pixel = [103.939, 116.779, 123.68]

    X_train = X_train.transpose((0, 3, 1, 2))
    X_val = X_val.transpose((0, 3, 1, 2))

    for c in range(3):
      print 'subtracting {c}'.format(c=mean_pixel[c])
      X_train[:, c, :, :] = X_train[:, c, :, :] - mean_pixel[c]
      X_val[:, c, :, :] = X_val[:, c, :, :] - mean_pixel[c]


    perm = permutation(len(y_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    y_train = map(int, [x.replace('c', '') for x in y_train])
    y_val = map(int, [x.replace('c', '') for x in y_val])

    y_train = np_utils.to_categorical(y_train, 10)
    y_val = np_utils.to_categorical(y_val, 10)

    print X_train.shape, y_train.shape
    print X_val.shape, y_val.shape

    print 'creating model'
    model = vgg_std16_model(img_rows, img_cols)

    print 'fitting model'
    model.fit(X_train, y_train, batch_size=batch_size,
            nb_epoch=nb_epoch,
            verbose=1,
            validation_data=(X_val, y_val),
            shuffle=True)

    f_test = h5py.File(os.path.join(cache_path, 'test_224_224.h5'),'r')

    X_test_id = np.array(f_test['X_test_id'])

    print 'size of test'
    print len(X_test_id)
    print X_test_id.shape

    print 'subtracting mean'

    mean_pixel = [103.939, 116.779, 123.68]

    print 'predicting'

    preds = []
    iter_size = 2 * 4096
    for i in xrange(0, num_images, iter_size):
      start_i = i
      end_i = min(num_images, i + iter_size)
      print start_i, end_i

      X_test = np.array(f_test['X_test'][start_i:end_i], dtype=np.float32)
      X_test = X_test.transpose((0, 3, 1, 2))
      print X_test.shape
      for c in range(3):
        print 'subtracting {c}'.format(c=mean_pixel[c])
        X_test[:, c, :, :] = X_test[:, c, :, :] - mean_pixel[c]

      X_test = X_test.astype(np.float32)

      preds += [model.predict(X_test, batch_size=32, verbose=1)]

    predictions = np.vstack(preds)

    print 'saving result'
    result = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                   'c4', 'c5', 'c6', 'c7',
                                                   'c8', 'c9'])

    result['img'] = X_test_id


    if not os.path.isdir('subm'):
        os.mkdir('subm')


    sub_file = os.path.join('subm', '{ind}_submission_'.format(ind=ind) + suffix + '.csv')
    result.to_csv(sub_file, index=False)
    ind += 1


    f_test.close()

  f.close()

