/* All rights reserved
   Created by: Anuj Shrivatsav Shrikanth and Anush Sriram Ramesh
   EECE 5639 Computer Vision project - Stereo-Vision
   Main file to call functions from stereoVisionFunc.cpp in <root_of_repo>/inc/stereoVisionFunc.hpp */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../inc/stereoVisionFunc.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // create a StereoVision object
    StereoVision stereoVision;
    // create a cv_factory object
    cv_factory cvFactory;
    // create a vector of strings to store the names of the images
    vector<string> imageNames;
    // push the names of the images into the vector
    imageNames.push_back("cartoon/left.jpeg");
    imageNames.push_back("cartoon/right.jpeg");
    // read the image
    vector<Mat> stereoImages = cvFactory.read_images(imageNames);
    // find feature points in the image
    vector<Point2f> leftPoints = stereoVision.findFeaturePoints(stereoImages[0]);
    vector<Point2f> rightPoints = stereoVision.findFeaturePoints(stereoImages[1]);
    // draw feature points on the image
    Mat leftImageWithPoints = stereoVision.drawFeaturePoints(stereoImages[0], leftPoints);
    Mat rightImageWithPoints = stereoVision.drawFeaturePoints(stereoImages[1], rightPoints);
    // save the image with feature points marked on it to the ../output directory
    cvFactory.saveImage("../output/leftImageWithPoints.jpg", leftImageWithPoints);
    cvFactory.saveImage("../output/rightImageWithPoints.jpg", rightImageWithPoints);
}