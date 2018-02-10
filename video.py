################################################################################
# code sources:
#   * Udacity project lecture
#   * Udacity Q&A project video (https://www.youtube.com/watch?v=P2zwrTM8ueA)
#   * idea for averaging over last 25 frames from:
#     https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/mobilenet-ssd-predict.ipynb
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
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from functions import *
from scipy.ndimage.measurements import label

# load the test images
images = glob.glob('test_images/*.jpg')
# images = ['test_images/test3.jpg']
# print(images)

################################################################################
# pipeline for processing video
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
def process_img(img):

    # global variable to avg boxes
    global recent_boxes

    # generate windows with multiscale approach
    # for scale in [0.75, 1.5, 2, 3]:
    for scale in [0.75, 1.5, 2, 3]:
        raw_img, boxes = find_cars(img, y_start_stop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        recent_boxes.append(boxes)

    # averaging over 25 frames of the input video
    if (len(recent_boxes) > 25):

        # generate list of boxes to be averaged
        avg_boxes = []
        recent_boxes = recent_boxes[-25:]
        for boxlist in recent_boxes:
            for box in boxlist:
                avg_boxes.append(box)

        # initialize heat map and add heat
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = add_heat(heat, avg_boxes)

        # filter out false positives
        thresh_heat = apply_threshold(heat, 20)

        # visualize the heatmap when displaying
        heatmap = np.clip(thresh_heat, 0, 255)

        # process labels and draw image
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)

        return draw_img
    # for the first second of the video do not draw boxes
    else:
        return img


################################################################################
# video pipeline
################################################################################

output_video = 'project_output.mp4'
input_video = 'project_video.mp4'
# input_video = 'test_video.mp4'

# global variable for found car boxes
recent_boxes = []

clip1 = VideoFileClip(input_video)
# clip1 = clip1.subclip(38,42)
# clip1 = clip1.subclip(19,23)
video_clip = clip1.fl_image(process_img)
video_clip.write_videofile(output_video, audio = False)
