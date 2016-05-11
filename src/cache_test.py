from __future__ import division
'''
file reads images and caches them into h5 file with keys:
X_test: numpy array with shape (, 224, 224, 3) which is consistent with VGG16 input shape
fName: numpy array with shape (,) which defines filename of each image

based on https://github.com/ZFTurbo/KAGGLE_DISTRACTED_DRIVER/blob/master/run_keras_simple.py
'''
import os
import h5py
import glob
import cv2
import numpy as np
import math

def get_im(path, img_rows, img_cols):
    img = cv2.imread(path)
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

def load_test(img_rows, img_cols):
    print('Read test images')
    path = os.path.join('..', 'data', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files) / 10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id

if __name__ == '__main__':
	data_root = os.path.join(os.path.expanduser('~'), 'workspace', 'kaggle_statefarm', 'data')
	cache_root = os.path.join(os.path.expanduser('~'), 'workspace', 'kaggle_statefarm', 'src', 'cache')

	img_rows, img_cols = 224, 224
	X_test, X_test_id = load_test(img_rows, img_cols)
	X_test = np.array(X_test, dtype=np.uint8)

	f = h5py.File(os.path.join(cache_root, 'test_224_224.h5'),'w')
	f['X_test'] = X_test
	f['X_test_id'] = X_test_id
	f.close()