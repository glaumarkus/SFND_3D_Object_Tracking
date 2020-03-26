## SensorFusion: Collision Detection System

<p align="center">
  <img src="/media/FAST-BRIEF.gif" alt="result"
  title="result"  />
</p>

The goal of this project is to create a collision detection system, based on camera and lidar sensors. With the relative movement of the preceeding vehicle in front of us, we are able to calculate a Time-To-Collision (TTC), assuming a constant velocity model. This is obviously not correct, since there are some factors in place that will cause our prediction to be off. 

1. Own Movement is not constant
2. Preceeding vehicles movement is not constant

Additionally there are also some challenges with handling the input data. It consists of 19 Prefiltered images from the [Kitti dataset](http://www.cvlibs.net/datasets/kitti/) with corresponding LIDAR point clouds. Data is loaded into a buffer of length=2 to calculate TTC.

<p align="center">
  <img src="/media/Raw_Image_data.gif" alt="result"
  title="result"  />
</p>

To link observations between 2 frames, a standard detection framework (YOLO) is used. It returns the most likely bounding box of an object, for example a car from the image input. The LIDAR data is then loaded from file. This data gets cropped for the region of interest, 20m in front and only the ego lane with a width of 4 meters. To associate LIDAR points with individual vehicles, field of view from camera is calculated and used to filter remaining LIDAR points. 

The image processing uses a combination of detectors and describers already explained in the preceeding project. The resulting keypoints are then matched across 2 frames. The bounding box from YOLO detector is then used to associate the previous bounding box ID with the current bounding box ID. Since the amount of bounding boxes is usually below 10 in total I went with a naive solution and just compared the amount of keypoints within the bounding boxes of the last and current frame. The maximum amount of keypoints contained is highly likely to be the correct bounding box ID in the current frame. 

### Performance Evaluation LIDAR-based TTC

With the filtered LIDAR data I am already set to calculate TTC based on this information. To evaluate later performance of only camera based calculations, I will assume that LIDAR data is the ground truth of TTC. So to get a small glimpse of the situation, let's take a look into observed distances and relative speed of the preceeding vehicle:

<p align="center">
  <img src="/media/ground_truth_pos.png" alt="result"
  title="result"  />
</p>

As can be seen, the distance to the preceeding vehicle decreases in a more or less linear way. Unfortunatly the relative speed to the vehicle is everything but smooth. It ranges from ~-0.3m/s up to ~1.2m/s, which will have a significant impact on the calculated TTC. To smooth this down, I applied also a buffer for TTC calculation, which takes the mean of last 2 calculated TTC's. This is a simple method to decrease the variance of the observations and obtain a more stable result. Generally the computation is done by retrieving the smallest point along the x-axis of LIDAR data that is part of a cluster. Therefore all points will need to be clustered using a tree. I set a really high min_size of 150, so other obstacles on the road dont get mistaken for preceeding vehicles. After retrieving both closest x-distances from the last and current frame, the TTC can be calculated by applying constant velocity model with regard to the framerate. 

<p align="center">
  <img src="/media/smooth_lidar.png" alt="result"
  title="result"  />
</p>

For camera TTC calculation, we will use the identified keypoints within each frame. Since the rectangular bounding box does not actually reflect the dimensions of a car, the keypoints and their matches will need to be filtered. This means to remove any keypoints that are on the road or in the sky/background with a high certainty. In general we would expect each keypoint on the car to move a certain similar distance between 2 observations, therefore assuming gaussian-like distances. High distances would generally mean that the match is wrong, really low distances that its likely part of the sky/background which does not move at all. My simple approach is just to calculate the mean and only keep keypointmatches with a distance <= standard deviation. 

For the actual TTC calculation a set of lines between all keypoint matches in both frames are computed, where the min length of one line is at least 80px. With both observations and the length of their distance a ratio between current & previous observations can be calculated and stored. I tried out both the mean and median with the stored ratios and got more stable results with the median. 

Within the provided code template I made some adjustments to track the performance of all detector/descriptor combinations. I read all valid combinations from file and run the script with the current combination. Additionally I save every image generated instead of displaying it. This way I can also take a look later and make some visualizations from them. Since the process takes quite some time to finish (with 30 combinations), this can be disabled by simply removing all combinations from the according file (combinations.txt). Additionally to receiving image output, I store the performance of all combinations within a "performance.csv" file. 

So lets take a look in the performance of camera based TTC in comparison to LIDAR.

### Performance Evaluation Camera-based TTC

First objective when comparing all different combinations is filtering out some combinations that did not work well on the given example. Comparing all combinations, it can be seen that 'HARRIS-FREAK', 'HARRIS-BRISK', 'HARRIS-ORB', 'HARRIS-BRIEF', 'HARRIS-SIFT', 'FAST-SIFT', 'ORB-FREAK', 'ORB-ORB', 'ORB-BRISK' and 'ORB-SIFT' were not able to identify enough keypoint matches across all frames or had infinities as TTC estimate and are therefore dropped in the analysis. 

<p align="center">
  <img src="/media/count_ttc.png" alt="result"
  title="result"  />
</p>

To ease up the visual performance analysis, I will provide an "Error per Frame" chart, as well as a boxplot to analyse the errors with a quick glance. The error per frame is the absolute error compared to the LIDAR measurement. Augmenting the Camera measurement for analysis is advantageous, so that there is always a reference visible. 

<p align="center">
  <img src="/media/AKAZE.png" alt="result"
  title="result"  />
</p>
<p align="center">
  <img src="/media/BRISK.png" alt="result"
  title="result"  />
</p>
<p align="center">
  <img src="/media/FAST.png" alt="result"
  title="result"  />
</p>
<p align="center">
  <img src="/media/ORB.png" alt="result"
  title="result"  />
</p>
<p align="center">
  <img src="/media/SHITOMASI.png" alt="result"
  title="result"  />
</p>
<p align="center">
  <img src="/media/SIFT.png" alt="result"
  title="result"  />
</p>

As can be seen in the plots above SHITOMASI and FAST both perform really well and have a small error compared to other results. Overall good combinations would be:
. Shitomasi - BRIEF
. Shitomasi - FREAK
. FAST - BRIEF
. FAST - ORB

Also considering the runtime analysis, which I featured in my [2D Feature Tracking project](https://github.com/glaumarkus/Camera-2D-Feature-Tracking), my overall recommendation for usage in an embedded system would be:
- FAST - ORB
- FAST - BRIEF

Both could most certainly run within an embedded system with 10 fps since they both only require around 5ms each.

#### FAST - ORB 
<p align="center">
  <img src="/media/FAST-ORB.gif" alt="result"
  title="result"  />
</p>

#### FAST - BRIEF
<p align="center">
  <img src="/media/FAST-BRIEF.gif" alt="result"
  title="result"  />
</p>


## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.
