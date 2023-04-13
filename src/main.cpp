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
    vector<Point> leftPoints = stereoVision.findFeaturePoints(stereoImages[0]);
    vector<Point> rightPoints = stereoVision.findFeaturePoints(stereoImages[1]);
    // find the correspondances between the feature points in the left and right images
    vector<pair<Point, Point>> correspondances = cvFactory.find_correspondences(stereoImages[0], stereoImages[1], leftPoints, rightPoints);
    // best fundamental matrix
    Mat bestFundamentalMatrix;
    // estimate the best correspondences
    // vector<pair<Point, Point>> bestCorrespondances = stereoVision.bestCorrespondences(correspondances, bestFundamentalMatrix);
    vector<pair<Point, Point>> bestCorrespondances = stereoVision.bestFundamentalMatrix(correspondances, bestFundamentalMatrix);
    // draw the correspondances between the feature points in the left and right images
    Mat correspondancesImage = cvFactory.draw_lines(stereoImages[0], stereoImages[1], bestCorrespondances);
    // draw feature points on the image
    Mat leftImageWithPoints = stereoVision.drawFeaturePoints(stereoImages[0], leftPoints);
    Mat rightImageWithPoints = stereoVision.drawFeaturePoints(stereoImages[1], rightPoints);
    // compute the dense disparity map
    Mat disparityMap = stereoVision.computeDisparityMap(stereoImages[0], stereoImages[1], bestFundamentalMatrix);
    // save the image with feature points marked on it to the ../output directory
    cvFactory.saveImage("../output/leftImageWithPoints.jpg", leftImageWithPoints);
    cvFactory.saveImage("../output/rightImageWithPoints.jpg", rightImageWithPoints);
    cvFactory.saveImage("../output/correspondancesImage.jpg", correspondancesImage);
    cvFactory.saveImage("../output/disparityMap.jpg", disparityMap);

    // stereoVision.sample(stereoImages[0], stereoImages[1] );
}