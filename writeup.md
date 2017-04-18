##Writeup

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./test_images/carandnotcar.png
[image2]: ./test_images/somehog.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./test_images/6.png
[image8]: ./test_images/1.png
[image9]: ./test_images/2.png
[image10]: ./test_images/3.png
[image11]: ./test_images/4.png
[image12]: ./test_images/5.png
[video1]: ./output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed some of them to get a feel for what the `skimage.hog()` output looks like. 

The code for this step is contained in the second code cell of the IPython notebook.  

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and after train the SVM classifier again and again. I find the best HOG parameters is `color_space = 'YCrCb'`,`hog_chanel = 'All'`,`orient = 6`,
`pix_per_cell = 8`,
`cell_per_block = 3`.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using color features and hist features. The code for this step is contained in the No.3~No.6 code cells of the IPython notebook. use these features the test accuracy is just 89% so I move on to HOG features

The SVM with HOG features is much more accuracy. test accuracy is 98%. The code for this step is contained in the No.7 code cell.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions at fixd scales all over the image. I also cut out the sky and others that not possiblely contain cars. then I get the HOG features of these image and apply them to SVM classifier I just trained. if it is a car I will put a blue cube on it's location. the code is in the No.9 and No.10 cells of the notebook.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched the windows using YCrCb 3-channel HOG features only in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap that contain the positive detections of series of frames(5 of them) and then thresholded that map to identify vehicle positions. 

	heat_list[0] = heat_list[1]
	heat_list[1] = heat_list[2]
	heat_list[2] = heat_list[3]
	heat_list[3] = heat_list[4]
	heat_list[4] = heat
	
	for i in range(4):
	    heat = heat + heat_list[i]
	
	heat = apply_threshold(heat,10)

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
the code of this step is in the code cell no.11 and No.14

### Here the resulting bounding boxes:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

first is the SVM classifier. I think this is not the best approach to cope with the image classification problem. later on I will try CNN and I believe I will get much more accuracy.

Because the SVM classifier is a little bit overfitting  I get a lot of false positives in my video pipeline. So I apply the heatmap methord to several frame of the videos. Considering all of them and get rid of the false positives that only appear in some frames.

The heatmap approach can reduce the amount of false positives but it's also make the algorithm can't deal with the instance situation(such as a very fast lamborghini flew by in light speed and the algorithm thinks that is just a false positive). so I think the way to imporve is still make a better classifier.