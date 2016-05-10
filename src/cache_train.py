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
import pandas as pd
from numpy.random import permutation

def get_im(path, img_rows, img_cols):
    img = cv2.imread(path)
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

if __name__ == '__main__':
    data_root = os.path.join(os.path.expanduser('~'), 'workspace', 'kaggle_statefarm', 'data')


    img_rows, img_cols = 224, 224

    print 'saving'
    f = h5py.File(os.path.join(data_root, 'train_224_224.h5'),'w')

    drivers = pd.read_csv(os.path.join(data_root, 'driver_imgs_list.csv'))
    for driver in drivers['subject'].unique():
        print driver
        drivers_subset = drivers[drivers['subject'] == driver]
        X = []
        y = []
        for i in drivers_subset.index:
            img = get_im(os.path.join(data_root,
                'train',
                drivers.loc[i, 'classname'],
                drivers.loc[i, 'img']), img_rows, img_cols)
            X += [img]
            y += [drivers.loc[i, 'classname']]


        f['X_{driver}'.format(driver=driver)] = X
        f['y_{driver}'.format(driver=driver)] = y
    f.close()
