## **Vehicle Detection Project**

Once again, without all of the lectures and quizes in this module, I would not have stood a chance. I used a couple of different files to complete this project. I used the notebook titled, "P5test.ipynb" to do my initial work and picture grabbing and a second titled, "P5Video.ipynb" to create my video output file named, "myVideo_finalt1RBF.mp4". 

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier.
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/yChannelHog.png "Y-Channel Car"
[image2]: ./output_images/crChannelHog.png "y-Channel Not Car"
[image3]: ./output_images/cbChannelHog.png "Cb-Channel"
[image4]: ./output_images/myGrid.png "My Search Grid"
[image5]: ./output_images/preHeatMap.png "Pre-Heat Map"
[image6]: ./output_images/heatMapped.png "Post-Heat Map"
[image7]: ./examples/heatMapped.png "Post Heat Map"
[video1]: ./myVideo_finalt1RBF.mp4
[video2]: ./myVideo_finalt2LIN.mp4
[video3]: ./myVideo_ma5t3LIN.mp4



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
# Writeup / README

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

# Histogram of Oriented Gradients (HOG)

1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the 'P5Video' IPython notebook, named "Step 2: Choose and Train Colorspace and SVM."  The method for this was taken directly from lecture. 

I started by reading in all the `vehicle` and `non-vehicle` images. 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed an image from the car class and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. The images are of a car and a non-car in the 'Y' colorspace hogged:

![alt text][image1]

![alt text][image2]



2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of colorspaces and parameters. The parameter testing came after I slected a classifier. I ended up on settling on all the three channels from the YCrCb colorspace. I know in a real situation using all three color channels is probably not going to work, because it is about speed and accuracy. For this project I tried to focus more on accuracy, because I figured speed would come more easily if I knew what was going on. Using one color channel versus three dropped my 'time to extract' from 52 seconds to 26 seconds for my training data. Therefore, I could probably count on HOG extraction to be about twice as fast if I use one channel versus three. I did my colorspace testing using a Linear SVC model. Looking back, this is not a good scientific way to do this. I am sure each colorspace works better with different classifiers. The problem was that accuracy numbers were so good, that I got carried away without using the scientific method. Shown below is some of my testing:

| Colorspace      | Accuracy   | Model   | Time to Classify 10 items (s) |  
|:-------------:|:-------------:| :-------------:| :-------------:| 
| Grayscale      | 0.9505        | LinearSVC(C=1) | .01701 |
| RGB      | 0.9675        | LinearSVC(C=1) | .016021 |
| HSV      | 0.9818       | LinearSVC(C=1) | .01301 |
| LUV      | 0.9843        | LinearSVC(C=1) | .01802 |
| HLS     | 0.9843        | LinearSVC(C=1) | .01301 |
| YUV     | 0.9872       | LinearSVC(C=1) | .01301 |
| YCrCb      | 0.9875       | LinearSVC(C=1) | .01301 |


After I decided on the YCrCb colorspace, I played with the orientation, pix per cell, cells per block, but had the best success with the 9, 8, 2 setup that was shown to us in lecture.

3. Describe how (and identify where in your code) you trained a classifier.

I trained a classifer using my Orientation = 9, Pix/Cell = 8, and Cell/Block = 2 HOG features . I chose an SVC with an RBF kernel(C=10), just because it had the highest accuracy when tested and it made the nicest video. I know that his is not a real plausible solution after seeing how slow it ran very, very slow. Below is a table I used to help me:

| Colorspace      | Accuracy   | Model   | Time to Classify 10 items (s) |  
|:-------------:|:-------------:| :-------------:| :-------------:| 
| YCrCb      | 0.9875       | LinearSVC(C=1) | .01301 |
| YCrCb      | 0.9843       | SVC(C=.1) - RBF | .2011 |
| YCrCb      | 0.9937       | SVC(C=1) - RBF| .1381 |
| YCrCb      | 0.9949      | SVC(C=10) - RBF | .1401 |

I also trained a LinearSVC, but didn't really like how the thresholding and video turned out. It was much faster, but wasn't as nice. I know this is a balance, but I lost out on speed.

## Sliding Window Search

1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions at fixed scale and location. I made a giant, oversized grid. I think I ended up with 488 different boxes to search. I went with three layers of boxes, because one wasn't cutting it. I also used rectangular boxes instead of square, because that is how the car looked to me. I honestly did a bunch of 'guess and check' over the test images until I found a solution that worked. The code for this can be found in the fifth code cell of the 'P5Video' IPython notebook, named "Step 3: Create Pipeline and Function Classes."  Below is an example an image of search grid:

![alt text][image4]

2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on all of the YCrCb 3-channel HOG features. To improve the reliability of the classifier i.e., fewer false positives and more reliable car detections I made sure I scalled the HOG features that I found and also set the feature_vec of the hog function to true. This "returns the data as a feature vector by calling .ravel() on the result just before returning". The goal was to input video data the same way as the SVC trained.

### Video Implementation

1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here are my three best from my different ideas;

#SVC(RBF, C=10):
Here's a [link to my video result](./myVideo_finalt1RBF.mp4)


#LinearSVC without Moving Average:
Here's a [link to my video result](./myVideo_finalt2LIN.mp4)

#LinearSVC with Moving Average:
Here's a [link to my video result](./myVideo_ma5t3LIN.mp4)

2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

I used the heat mapping threshold idea shown in lecture. The code for this can be found in the sixth code cell of the 'P5Video' IPython notebook, named "Step 3: Create Pipeline and Function Classes." I also implemented a moving average of the heat maps, but that idea, which I thought was awesome, turned out not to work so well. Here is an example image before and after heat mapping was applied:

#Pre-Heat Map

![alt text][image5]

#Post-Heat Map

![alt text][image6]
---

---

### Discussion

1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The amount of improvements that could be made to my project are endless. There is literally not enough internet to hold everything that probably should be fixed. Using the YCrCb colorspace combinded with the SVC(RBF kernel) worked well, but is very, very slow and probably could never be used in any real-time application. Also, I sort of cheated myself by not looking into spatially binned color and histograms of color in the feature vector, because my inital pass through had excellent results. So I think those could probably add some value. Since I am on the topic of speed, my 'grid' needs far less windows to search. This eats a lot of time and with enough tinkering, I think I could find a more optimal search pattern.

My video has some misses in detecting cars, so maybe a heat threshold isn't all we need. I tried to get a moving average to work, because it made sense in my head, but I think a more sophisticated filter may be needed. As with any classifier, I could train it using more data. The more data the better. Also, instead of using the classic '.predict' I would like to see how the 'decision_function' could be used for SVCs.

The biggest thing that I took away from this project is that I need to start thinking more like a scientist rather then an engineer. There are so many knobs that are available to turn that each one, from colorspace to threshold, needs to be carefully looked at. 

