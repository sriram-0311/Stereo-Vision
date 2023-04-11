/* All rights reserved
   Created by : Anuj Shrivatsav Shrikanth and Anush Sriram Ramesh
   EECE 5639 Computer Vision project - Stereo-Vision */

/* Explore Stereo Vision */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <numeric>
#include <filesystem>
#include <math.h>
#include <cv_factory.hpp>

using namespace std;
using namespace cv;

class StereoVision {
    public:
        // function to find feature points in the image using SIFT algorithm and return pair of points
        vector<Point> findFeaturePoints(Mat img1) {
            cout<<">>> findFeaturePoints() called"<<endl;
            // create SIFT object
            Ptr<SIFT> sift = SIFT::create();
            // create a vector of keypoints
            vector<KeyPoint> keypoints;
            // create a Mat object to store the descriptors
            Mat descriptors;
            // detect keypoints and compute descriptors
            sift->detectAndCompute(img1, noArray(), keypoints, descriptors);
            // create a vector of points
            vector<Point> points;
            // push the points into the vector
            for (int i = 0; i < keypoints.size(); i++) {
                points.push_back(keypoints[i].pt);
            }
            cout<<"<<< findFeaturePoints() returned"<<endl;
            // return the vector of points
            return points;
        }

        // function to take one image as input and feature points for that image and return the image with feature points marked on it
        Mat drawFeaturePoints(Mat img1, vector<Point> points) {
            cout<<">>> drawFeaturePoints() called"<<endl;
            // create a Mat object to store the image with feature points marked on it
            Mat img1_with_points = img1.clone();
            // draw the feature points on the image
            for (int i = 0; i < points.size(); i++) {
                circle(img1_with_points, points[i], 2, Scalar(0, 0, 255), 1);
            }
            // return the image with feature points marked on it
            cout<<"<<< drawFeaturePoints() returned"<<endl;
            return img1_with_points;
        }

        // function to take eight corresponding points as input and return the estimated fundamental matrix
        Mat estimateFundamentalMatrix(vector<Point> points1, vector<Point> points2) {
            cout<<">>> estimateFundamentalMatrix() called"<<endl;
            // create a Mat object to store the fundamental matrix
            Mat fundamentalMatrix;
            // create a vector of points to store the corresponding points
            vector<Point2f> points1_2f, points2_2f;
            // push the corresponding points into the vector
            for (int i = 0; i < points1.size(); i++) {
                points1_2f.push_back(Point2f(points1[i].x, points1[i].y));
                points2_2f.push_back(Point2f(points2[i].x, points2[i].y));
            }
            // find the fundamental matrix using the eight corresponding points
            fundamentalMatrix = findFundamentalMat(points1_2f, points2_2f, FM_8POINT);
            // return the fundamental matrix
            cout<<"<<< estimateFundamentalMatrix() returned"<<endl;
            return fundamentalMatrix;
        }

        // function to take corresponding points as input and return the best inliers correspondences using RANSAC algorithm by estimating the fundamental matrix for 50 iterations
        vector<pair<Point, Point>> bestCorrespondences(vector<pair<Point, Point>> correspondingPoints, Mat& bestFundamentalMatrix) {
            cout<<">>> bestCorrespondences() called"<<endl;
            // create a vector of points to store the corresponding points
            vector<Point> points1, points2;
            vector<pair<Point, Point>> tempBestCorrespondences;
            vector<pair<Point, Point>> bestCorrespondences;
            int maxInliers = 0;
            // for 50 iterations
            for (int i = 0; i < 50; i++) {
                // clear the vectors
                points1.clear();
                points2.clear();
                // push 8 corresponding points into the vector
                for (int j = 0; j < 8; j++) {
                    int index = rand() % correspondingPoints.size();
                    points1.push_back(correspondingPoints[index].first);
                    points2.push_back(correspondingPoints[index].second);
                }
                // find the fundamental matrix using the eight corresponding points
                Mat fundamentalMatrix = estimateFundamentalMatrix(points1, points2);
                // print the fundamental matrix
                // cout<<"Fundamental Matrix: "<<endl<<fundamentalMatrix<<endl;
                int inliers = 0;
                tempBestCorrespondences.clear();
                // transform all the corresponding points using the fundamental matrix
                for (int j = 0; j < correspondingPoints.size(); j++) {
                    Mat point1 = (Mat_<double>(3, 1) << correspondingPoints[j].first.x, correspondingPoints[j].first.y, 1);
                    Mat point2 = (Mat_<double>(3, 1) << correspondingPoints[j].second.x, correspondingPoints[j].second.y, 1);
                    // Mat point1_transpose = point1.t();
                    // // print the shape of the point
                    // cout<<"Shape of point: "<<point1.size()<<endl;
                    // Mat point2_transpose = point2.t();
                    Mat epipolarLine1 = fundamentalMatrix * point2;
                    Mat epipolarLine2 = fundamentalMatrix.t() * point1;
                    // calculate the distance of the corresponding points from the epipolar lines
                    double distance1 = abs(point2.dot(epipolarLine1)) / sqrt(pow(epipolarLine1.at<double>(0, 0), 2) + pow(epipolarLine1.at<double>(1, 0), 2));
                    double distance2 = abs(point1.dot(epipolarLine2)) / sqrt(pow(epipolarLine2.at<double>(0, 0), 2) + pow(epipolarLine2.at<double>(1, 0), 2));
                    // if the distance is less than 0.1, then the corresponding points are inliers
                    if (distance1 < 0.3 && distance2 < 0.3) {
                        inliers++;
                        tempBestCorrespondences.push_back(correspondingPoints[j]);
                    }
                }
                // if the number of inliers is greater than the maximum number of inliers, then update the maximum number of inliers and the best correspondences
                if (inliers > maxInliers) {
                    maxInliers = inliers;
                    bestCorrespondences = tempBestCorrespondences;
                    bestFundamentalMatrix = fundamentalMatrix;
                }
            }
            // return the best correspondences
            cout<<"<<< bestCorrespondences() returned"<<endl;
            return bestCorrespondences;
        }

             // function to compute the dense disparity map using best fundamental matrix to reduce the search space

        // Compute a dense disparity map using the Fundamental matrix to help reduce the search space. The output is 3 images
        // 1. Vertical disparity component
        // 2. Horizontal disparity component
        // 3. Disparity vector using color, where the direction of the vector is coded by hue, and the length of the vector is coded by saturation
        // For displaying grayscale images scale the intensities to the range 0-255

        Mat computeDisparityMap(Mat img1, Mat img2, Mat bestFundamentalMatrix) {
            cout<<">>> computeDisparityMap() called"<<endl;
            // create 3 Mat objects to store the disparity map for each component
            Mat disparityMapVertical = Mat::zeros(img1.rows, img1.cols, CV_8UC1);
            Mat disparityMapHorizontal = Mat::zeros(img1.rows, img1.cols, CV_8UC1);
            int minDisparityHorizontal = 5;
            int minDisparityVertical = 5;
            int windowSize = 5;
            // pad the images with zeros
            Mat img1_padded = Mat::zeros(img1.rows + 2 * windowSize, img1.cols + 2 * windowSize, CV_8UC1);
            Mat img2_padded = Mat::zeros(img2.rows + 2 * windowSize, img2.cols + 2 * windowSize, CV_8UC1);
            // create a Mat object to store the epipolar line of the second image
            Mat epipolarLine;
            // create a Mat object to store the distances of the difference in pixels between image 1 and 2 in grayscale
            Mat disparityMap = Mat::zeros(img1.rows, img1.cols, CV_8UC1);
            // create a Mat object to store the epipolar line of the second image
            vector<Point2d> epipolarLine2;
            for (int i = 0; i < 1; i++) {
                for (int j = 0; j < 1; j++) {
                    Mat point1 = (Mat_<double>(3, 1) << i, j, 1);
                    // extract a 5x5 window around the pixel
                    Mat window1 = img1_padded(Rect(j, i, 5, 5));
                    epipolarLine = point1.t()*bestFundamentalMatrix;
                    for (int k = 0; k < img2.rows; k++) {
                        for (int l = 0; l < img2.cols; l++) {
                            Mat point2 = (Mat_<double>(1, 3) << l, k, 1);
                            // calculate the distance of the corresponding points from the epipolar lines
                            // double distance = abs(epipolarLine.dot(point2)) / sqrt(pow(epipolarLine.at<double>(0, 0), 2) + pow(epipolarLine.at<double>(0, 1), 2));
                            double distance = abs(epipolarLine.dot(point2));
                            // store the distance in corresponding pixel in the disparity map
                            disparityMap.at<uchar>(k, l) = distance*255;
                            // if the distance is less than 0.1, then the corresponding points are inliers
                        //     if (distance < 0.001) {
                        //         // store the pixels in a vector that is the epipolar line
                        //         // change the point to a 2d point
                        //         Point2d point2(l, k);
                        //         epipolarLine2.push_back(point2);
                        //         // // calculate the disparity in both the horizontal and vertical directions
                        //         // int disparityHorizontal = abs(k - i);
                        //         // int disparityVertical = abs(j - l);
                        //         // if(disparityHorizontal < minDisparityHorizontal && disparityVertical < minDisparityVertical)
                        //         // {   minDisparityHorizontal = disparityHorizontal;
                        //         //     minDisparityVertical = disparityVertical;
                        //             // matchPoint = (Mat_<double>(3,1) << point2.at<double>(0), point2.at<double>(1), 1);}
                        //             // print the values in point 1 and match point
                        //             // cout<<"Point 1: "<<point1<<endl;
                        //             // cout<<"Match Point: "<<matchPoint<<endl;
                        //         }
                        //     // find the pixel correspondence in the second image by considering templates around the epipolar line
                        //     for (int m = 0; m < epipolarLine2.size(); m++) {
                        //         double bestDistance = 1000;
                        //         // extract a 5x5 window around the pixel
                        //         Mat window2 = img2_padded(Rect(epipolarLine2[m].x, epipolarLine2[m].y, 5, 5));
                        //         // find the pixel correspondence in the second image by considering templates around the epipolar line
                        //         cv_factory obj;
                        //         double distance = obj.NCC(window1, window2);
                        //         if(distance < bestDistance)
                        //         {
                        //             bestDistance = distance;
                        //             Mat matchPoint = (Mat_<double>(3,1) << epipolarLine2[m].x, epipolarLine2[m].y, 1);
                        //         }
                        //     }
                        // }
                        }
                    }
                }
            }
            // show the disparity map
            imshow("Disparity Map", disparityMap);
            waitKey(0);
            // return the disparity map
            cout<<"<<< computeDisparityMap() returned"<<endl;
            return disparityMapHorizontal;
        }
};
