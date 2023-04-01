# Stereo-Vision
Exploring stereo vision, by finding correspondences between left and right images. Computing the fundamental matrix for each pair and removing outliers by RANSAC. Compute Dense Disparity Map using the fundamental matrix. Displaying the output as 3 images - horizontal and vertial disparity maps and the third image representing the disparity vector using color where the direction is coded by the hue and length by the saturation. Scaling the disparity map to the range [0, 255] and saving the output as a grayscale image.

## Dependencies
* OpenCV 3.0
* CMake 2.8

## Description of files
* CMakeLists.txt: CMake file.
* main.cpp: Main function calling the functions to compute the disparity map.
* stereoVisionFunc.cpp: Contains the implementation of the functions to compute the disparity map.

## Build Instructions
* Clone this repo.
* Make a build directory: `mkdir build && cd build`
* Compile: `cmake .. && make`
* Run it: `./stereo_vision`.

## Input
* The input must be placed in the build folder. The input is a pair of images, left and right. The images must be named left.jpg and right.jpg.

## Output
* The output is saved in the output folder. The output is a grayscale image with the disparity map scaled to the range [0, 255].

## Methodology
* The input images are read and converted to grayscale.
* Find the keypoints and descriptors using SIFT detector.
* Match the descriptors using NCC implemented in ImageMosaicing project.
* Compute the fundamental matrix for each correspondence pair using the 8-point algorithm and eliminate the outliers using RANSAC.
* Display the inliers with left image above right image and save the image.
* Compute the dense disparity map using the fundamental matrix.
* Display the disparity map as 3 images - horizontal and vertial disparity maps and the third image representing the disparity vector using color where the direction is coded by the hue and length by the saturation.
* Scale the disparity map to the range [0, 255] and save the output as a grayscale image.

## Results
To be added...


