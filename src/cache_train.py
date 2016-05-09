from __future__ import division
'''
file reads images and caches them into h5 file with keys:
X_train: numpy array with shape (22424, 224, 224, 3) which is consistent with VGG16 input shape and layers order. Mean VGG16 values are also subrtacted [103.939, 116.779, 123.68]
y_train: numpy array with shape (22424) which defines class for each image
driver_id: array with strings defineing driver_ids

based on https://github.com/ZFTurbo/KAGGLE_DISTRACTED_DRIVER/blob/master/run_keras_simple.py
'''
import os
import h5py
import glob
import cv2
import numpy as np
from numpy.random import permutation

def get_driver_data():
    dr = dict()
    path = os.path.join('..', 'data', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr

def get_im(path, img_rows, img_cols):
    img = cv2.imread(path)
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

def load_train(img_rows, img_cols):
    X_train = []
    y_train = []
    driver_id = []

    driver_data = get_driver_data()

    print('Reading train images')
    for j in range(10):
        print('Load folder c{j}'.format(j=j))
        path = os.path.join('..', 'data', 'train',
                            'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im(fl, img_rows, img_cols)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    return X_train, y_train, driver_id

if __name__ == '__main__':
    data_root = os.path.join(os.path.expanduser('~'), 'workspace', 'kaggle_statefarm', 'data')

    img_rows, img_cols = 224, 224
    X_train, y_train, driver_id = load_train(img_rows, img_cols)
    X_train = np.array(X_train, dtype=np.uint8)
    X_train = X_train.transpose((0, 3, 1, 2))

    X_train = X_train.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]

    for c in range(3):
        print 'subtracting {c}'.format(c=mean_pixel[c])
        X_train[:, c, :, :] = X_train[:, c, :, :] - mean_pixel[c]

    y_train = np.array(y_train, dtype=np.uint8)

    print 'shuffling'
    perm = permutation(len(y_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    print 'saving'
    f = h5py.File(os.path.join(data_root, 'train_224_224.h5'),'w')

    f['X_train'] = X_train
    f['y_train'] = y_train
    f['driver_id'] = driver_id
    f.close()