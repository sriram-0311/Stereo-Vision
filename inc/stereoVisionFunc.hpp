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

        // function to compute the dense disparity map using best fundamental matrix
        Mat computeDisparityMap(Mat img1, Mat img2, Mat bestFundamentalMatrix) {
            cout<<">>> computeDisparityMap() called"<<endl;
            // compute the common region of the two images
            Rect commonRegion = computeCommonRegion(img1, img2, bestFundamentalMatrix);
            // create a Mat object to store the disparity map size of the common region
            Mat disparityMap(commonRegion.height, commonRegion.width, CV_8UC1);
            // for each pixel in the common region calculate the disparity in the horizontal direction and store it in the disparity map
            for (int i = 0; i < commonRegion.height; i++) {
                for (int j = 0; j < commonRegion.width; j++) {
                    int disparity = 0;
                    // for each pixel in the horizontal direction
                    for (int k = 0; k < img1.cols - commonRegion.x; k++) {
                        // if the pixel in the first image is not black and the pixel in the second image is not black
                        if (img1.at<Vec3b>(i + commonRegion.y, j + commonRegion.x)[0] != 0 && img1.at<Vec3b>(i + commonRegion.y, j + commonRegion.x)[1] != 0 && img1.at<Vec3b>(i + commonRegion.y, j + commonRegion.x)[2] != 0 && img2.at<Vec3b>(i + commonRegion.y, j + commonRegion.x + k)[0] != 0 && img2.at<Vec3b>(i + commonRegion.y, j + commonRegion.x + k)[1] != 0 && img2.at<Vec3b>(i + commonRegion.y, j + commonRegion.x + k)[2] != 0) {
                            // calculate the difference between the pixel in the first image and the pixel in the second image
                            int difference = abs(img1.at<Vec3b>(i + commonRegion.y, j + commonRegion.x)[0] - img2.at<Vec3b>(i + commonRegion.y, j + commonRegion.x + k)[0]) + abs(img1.at<Vec3b>(i + commonRegion.y, j + commonRegion.x)[1] - img2.at<Vec3b>(i + commonRegion.y, j + commonRegion.x + k)[1]) + abs(img1.at<Vec3b>(i + commonRegion.y, j + commonRegion.x)[2] - img2.at<Vec3b>(i + commonRegion.y, j + commonRegion.x + k)[2]);
                            // if the difference is less than 10, then the pixel in the first image and the pixel in the second image are similar
                            if (difference < 10) {
                                disparity = k;
                                break;
                            }
                        }
                    }
                    // store the disparity in the disparity map
                    disparityMap.at<uchar>(i, j) = disparity;
                }
            }
            // return the disparity map
            cout<<"<<< computeDisparityMap() returned"<<endl;
            return disparityMap;
        }
};
