from __future__ import division

'''
This script creates submission from a given model
'''

import numpy as np

import os
import glob
import cv2
import math
import h5py
import datetime
import pandas as pd
import sys
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


if __name__ == '__main__':
	cache_path = os.path.join(os.path.expanduser('~'), 'workspace', 'kaggle_statefarm', 'src', 'cache')

	model_weights = 'model_weightsVGG_32_20_.h5'
	model_architecture = 'architectureVGG_32_20_.json'

	model_path = os.path.join(os.path.expanduser('~'),
		'workspace',
		'kaggle_statefarm',
		'models')

	num_images = 79726

	batch_size = 32
	img_rows, img_cols = 224, 224

	f = h5py.File(os.path.join(cache_path, 'test_224_224.h5'),'r')


	X_test_id = f['X_test_id']


	print 'subtracting mean'

	mean_pixel = [103.939, 116.779, 123.68]


	print 'reading model'

	model = model_from_json(open(os.path.join(model_path, model_architecture)).read())
	model.load_weights(os.path.join(model_path, model_weights))

	model.compile(optimizer='SGD',
    	loss='categorical_crossentropy',
    	metrics=['accuracy'])

	print 'predicting'

	preds = []
	for i in xrange(0, num_images, batch_size):
		start_i = i
		end_i = min(num_images, i + batch_size)
		print start_i, end_i

		X_test = np.array(f['X_test'][start_i:end_i], dtype=np.uint8)
		X_test = X_test.transpose((0, 3, 1, 2))
		print X_test.shape
		for c in range(3):
		  print 'subtracting {c}'.format(c=mean_pixel[c])
		  X_test[:, c, :, :] = X_test[:, c, :, :] - mean_pixel[c]

		X_test = X_test.astype(np.float32)

		preds += [model.predict(X_test, batch_size=16, verbose=1)]

	predictions = np.vstack(preds)

	print 'saving result'
	result = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
	                                               'c4', 'c5', 'c6', 'c7',
	                                               'c8', 'c9'])

	result.loc[:, 'img'] = pd.Series(test_id, index=result.index)
	now = datetime.datetime.now()
	if not os.path.isdir('subm'):
	    os.mkdir('subm')

	suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
	sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
	result.to_csv(sub_file, index=False)

	f.close()