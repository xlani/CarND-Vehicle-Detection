################################################################################
# code sources:
#   * Udacity project lecture
#   * Udacity Q&A project video (https://www.youtube.com/watch?v=P2zwrTM8ueA)
################################################################################

################################################################################
# imports and load data
################################################################################

# imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from functions import *

################################################################################
# build a classifier and train it
################################################################################

# parameters to tune
colorspace = 'YCrCb' # can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # (32, 32)
hist_bins = 32
spatial_feat = True # spatial features on or off
hist_feat = True # histogram features on or off
hog_feat = True # HOG features on or off

# divide test images into cars and notcars
cars = glob.glob('./classifier_data/vehicles/**/*.png', recursive=True)
notcars = glob.glob('./classifier_data/non-vehicles/**/*.png', recursive=True)

print('Number of images with cars:', len(cars))
print('Number of images without cars:', len(notcars))

# reduce the sample size because HOG features are slow to compute
# sample_size = 1000
# cars = cars[0:sample_size]
# notcars = notcars[0:sample_size]

t=time.time()
car_features = extract_features(cars, cspace=colorspace, spatial_size=spatial_size,
                        orient=orient, hist_bins=hist_bins, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block, hog_channel=hog_channel)
notcar_features = extract_features(notcars, cspace=colorspace, spatial_size=spatial_size,
                        orient=orient, hist_bins=hist_bins, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block, hog_channel=hog_channel)
t2 = time.time()
print(round(t2-t, 2), 'seconds to extract HOG features...')

# create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

# fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)

# apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# use a linear SVC
# svc = LinearSVC()
svc = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)

# check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# check the prediction time for a single sample
t = time.time()
n_predict = 10
print('My SVC predicts:    ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels:', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'seconds to predict', n_predict,'labels with SVC')

# save the camera calibration result for later use
dist_pickle = {}
dist_pickle["svc"] = svc
dist_pickle["scaler"] = X_scaler
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins
pickle.dump(dist_pickle, open('svc_pickle.p', "wb"))
