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
from scipy.ndimage.measurements import label

# load the test images
images = glob.glob('test_images/*.jpg')
# images = ['test_images/test1.jpg']
# print(images)

################################################################################
# pipeline for processing images
################################################################################

# load saved classifier
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

# parameters
y_start_stop = [360, 680] # min and max in y to search in slide_window()

# for loop to go through the test images
for idx, fname in enumerate(images):

    # open image and init copies for drawing
    img = mpimg.imread(fname)
    window_img = np.copy(img)
    raw_imgs = np.copy(img)

    # initialize heat map
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # initialize list for found boxes
    box_list = []

    # generate windows with multiscale approach
    for scale in [0.75, 2, 1.5, 3]:

        raw_img, boxes = find_cars(img, y_start_stop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        # output images
        # window_img = draw_boxes(window_img, slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
        #                 xy_window=(int(64*scale), int(64*scale)), xy_overlap=(0.75, 0.75)), color=(0, int(255/3*scale),0), thick=2)
        raw_imgs = draw_boxes(raw_imgs, boxes)

        # save found boxes
        for box in boxes:
            box_list.append(box)
            # print(box_list)

    # Add heat to each box in box list
    heat = add_heat(heat, box_list)
    # filter out multiple detections and false positives
    tresh_heat = np.copy(heat)
    tresh_heat = apply_threshold(tresh_heat, 3)

    # visualize the heatmap when displaying
    heatmap = np.clip(tresh_heat, 0, 255)

    # process labels
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    print(labels[1], 'cars found in file', fname)

    # plot images
    mpimg.imsave('./output_images/'+fname[-9:-4]+'_cars_found.jpg', raw_imgs)
    mpimg.imsave('./output_images/'+fname[-9:-4]+'_heat.jpg', heat)
    mpimg.imsave('./output_images/'+fname[-9:-4]+'_tresh_heat.jpg', tresh_heat)
    mpimg.imsave('./output_images/'+fname[-9:-4]+'_processed.jpg', draw_img)
    # mpimg.imsave('./output_images/'+fname[-9:-4]+'_windows.jpg', window_img)
