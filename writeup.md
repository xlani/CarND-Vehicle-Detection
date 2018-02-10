###Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]:
[image3]: ./test_images/test1.jpg
[image4]: ./output_images/test1_windows.jpg
[image5]: ./output_images/test1_cars_found.jpg
[image6]: ./output_images/test1_heat.jpg
[image7]: ./output_images/test1_tresh_heat.jpg
[image8]: ./output_images/test1_processed.jpg
[video1]: ./project_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in in lines 41 through 61 of the file called `classifier.py`.  

I read in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters at the project sections 28 & 29 and stick with the following parameters (see lines 29 through 39 in file `classifier.py`):

```python
colorspace = 'YCrCb' # can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True # spatial features on or off
hist_feat = True # histogram features on or off
hog_feat = True # HOG features on or off
```

I managed to achieve an accuracy of around 99% on the test images.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM. For training I performed a split on the training image to seperate between training and test data. Before training the SVM I did normalizing and scaled the images correctly. The options for the linear SVC were extracted from the project section 35.

You can find the corresponding code in lines 69 through 107 in file `classifier.py`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to implement a sliding window search for four scale values `[0.75, 1.5, 2, 3]` in the bottom area of the image (see lines 62 through 75 in file `images.py`).

Here is an example image with windows drawn for all different search areas. Starting with `scale_factor = 0.75` going up to `3`, the windows borders getting brighter.

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are the most important steps for one test image:

Raw test image:
![alt text][image3]

Cars identified by classifier:
![alt text][image5]

Heat map:
![alt text][image6]

Heat map after thresholding (`tresh = 3`):
![alt text][image7]

Final result (single image):
![alt text][image8]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections of the last 25 frames I created a heatmap and then thresholded (`thresh = 30`) that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

See corresponding lines of code 53 through 92 in file `video.py`.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The SVM classifier detects in dark regions a lot of false positives. I tried to handle these with the 25-frames-approach and not by fine tuning the classifier. Nevertheless the classifier could be one thing to improve.

Another point could be implementing a neural network for car detection instead of the SVM classifier.
