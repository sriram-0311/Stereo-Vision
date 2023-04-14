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
        // create cv_factory object
        cv_factory cvFactory;
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

        // function to take two images and corresponding feature points as input and return the best fundamental matrix
        vector<pair<Point, Point>> bestFundamentalMatrix(vector<pair<Point, Point>> correspondingPoints, Mat& BestFundamentalMatrix) {
            cout<<">>> bestFundamentalMatrix() called"<<endl;
            // create 2 vectors to store the corresponding points in each image
            vector<Point> points1, points2;
            // push the corresponding points into the vectors
            for (int i = 0; i < correspondingPoints.size(); i++) {
                points1.push_back(correspondingPoints[i].first);
                points2.push_back(correspondingPoints[i].second);
            }
            // find the best fundamental matrix using cv::findFundamentalMat() function and ransac
            Mat inliers_mask;
            BestFundamentalMatrix = findFundamentalMat(points1, points2, FM_RANSAC, 0.01, 0.99, inliers_mask);
            // create a vector of pairs of points to store the best inliers correspondences
            vector<pair<Point, Point>> bestInliersCorrespondences;
            std::vector<Point2f> inliers1, inliers2;
            for (int i = 0; i < inliers_mask.rows; ++i) {
                if (inliers_mask.at<uint8_t>(i)) {
                    pair<Point, Point> bestInliersCorrespondence;
                    bestInliersCorrespondence.first = points1[i];
                    bestInliersCorrespondence.second = points2[i];
                    bestInliersCorrespondences.push_back(bestInliersCorrespondence);
                }
            }
            // return the best fundamental matrix
            cout<<"Best fundamental matrix is : "<<BestFundamentalMatrix<<endl;
            cout<<"<<< bestFundamentalMatrix() returned"<<endl;
            return bestInliersCorrespondences;
        }

        // function to take eight corresponding points as input and return the estimated fundamental matrix
        Mat estimateFundamentalMatrix(vector<Point> points1, vector<Point> points2) {
            cout<<">>> estimateFundamentalMatrix() called"<<endl;
            // create a Mat object to store the fundamental matrix
            Mat fundamentalMatrix;
            // construct the fundamental matrix empty
            fundamentalMatrix = Mat::zeros(3, 3, CV_32F);
            // number of points
            int num_points = points1.size();
            // create the A matrix to multiply with vectorized fundamental matrix
            Mat A(num_points, 9, CV_64FC1);
            for (int i = 0; i < num_points; i++) {
                double x1 = points1[i].x;
                double y1 = points1[i].y;
                double x2 = points2[i].x;
                double y2 = points2[i].y;
                A.at<double>(i, 0) = x2 * x1;
                A.at<double>(i, 1) = x2 * y1;
                A.at<double>(i, 2) = x2;
                A.at<double>(i, 3) = y2 * x1;
                A.at<double>(i, 4) = y2 * y1;
                A.at<double>(i, 5) = y2;
                A.at<double>(i, 6) = x1;
                A.at<double>(i, 7) = y1;
                A.at<double>(i, 8) = 1;
        }
        // Find SVD
        Mat U, S, Vt;
        SVD svd(A);
        Vt = svd.vt;
        Mat lastCol = Mat::zeros(1, 9, CV_64FC1);
        // convert the last column of Vt to a 3x3 matrix
        for (int i = 0; i < 9; i++)
            lastCol.at<double>(i) = Vt.at<double>(i, 8);
        cout<<"Vt :"<<lastCol.size()<<endl;
        // reshape the last column of Vt to get the fundamental matrix
        cout<<"Vt :"<<lastCol<<endl;
        Mat F = lastCol.reshape(0, 3);

        // Force F to be singular by setting smallest singular value to zero
        cv::SVDecomp(F, S, U, Vt);
        S.at<double>(2) = 0;
        F = U * Mat::diag(S) * Vt;

        // Unnormalize F
        // double norm_factor = 1.0 / F.at<double>(2, 2);
        // F *= norm_factor;
        // F.at<double>(2, 2) = 0.0;

        return F;
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
            for (int i = 0; i < 100; i++) {
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
                    Mat point1 = (Mat_<double>(1, 3) << correspondingPoints[j].first.x, correspondingPoints[j].first.y, 1);
                    Mat point2 = (Mat_<double>(1, 3) << correspondingPoints[j].second.x, correspondingPoints[j].second.y, 1);
                    // find the distance
                    Mat d = point1 * fundamentalMatrix * point2.t();
                    // convert the distance to double
                    double distance = d.at<double>(0, 0);
                    // // cout<<"d: "<<d<<endl;
                    // // if the distance is less than 0.01, then it is an inlier
                    if (abs(distance) < 0.001) {
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
            cout<<"Best Fundamental Matrix: "<<endl<<bestFundamentalMatrix<<endl;
            cout<<"Number of inliers: "<<maxInliers<<endl;
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
            // create a Mat object to store the disparity map and initialize it to zero with size of img1.cols + img2.cols and img1.rows
            Mat disparityMap = Mat::ones(img1.rows, img1.cols + img2.cols, CV_8UC1);
            int epipolarLineWindowSize = 10;
            Mat img1_gray, img2_gray;
            // convert the images to grayscale
            cvtColor(img1, img1_gray, COLOR_BGR2GRAY);
            cvtColor(img2, img2_gray, COLOR_BGR2GRAY);
            // for each pixel in the image 1
            for (int i = 0; i < img1.rows; i++) {
                for (int j = 0; j < img1.cols; j++) {
                    // create a Mat object to store the epipolar line
                    Mat epipolarLine = bestFundamentalMatrix * (Mat_<double>(3, 1) << j, i, 1);
                    // search along the epipolar line for the corresponding point in image 2 by performing NCC with a given window size
                    int disparity = 0;
                    double maxNCC = 0;
                    for (int k = 0; k < img2.cols; k++) {
                        // calculate the NCC
                        double ncc = 0;
                        double sum1 = 0, sum2 = 0, sum3 = 0;
                        for (int l = -epipolarLineWindowSize / 2; l <= epipolarLineWindowSize / 2; l++) {
                            for (int m = -epipolarLineWindowSize / 2; m <= epipolarLineWindowSize / 2; m++) {
                                if (i + l >= 0 && i + l < img1.rows && j + m >= 0 && j + m < img1.cols && k + l >= 0 && k + l < img2.rows && k + m >= 0 && k + m < img2.cols) {
                                    sum1 += img1_gray.at<uchar>(i + l, j + m) * img2_gray.at<uchar>(i + l, k + m);
                                    sum2 += pow(img1_gray.at<uchar>(i + l, j + m), 2);
                                    sum3 += pow(img2_gray.at<uchar>(i + l, k + m), 2);
                                }
                            }
                        }
                        ncc = sum1 / (sqrt(sum2) * sqrt(sum3));
                        // if the NCC is greater than the maximum NCC, then update the maximum NCC and the disparity
                        if (ncc > maxNCC) {
                            maxNCC = ncc;
                            disparity = k - j;
                        }
                    }
                    // store the disparity in the disparity map
                    // debug by printing the disparity
                    // cout<<"Disparity: "<<disparity<<endl;
                    disparityMap.at<uchar>(i, j) -= abs(disparity);
                }
            }
            // normalize the disparity map to the range 0-255
            normalize(disparityMap, disparityMap, 0, 255, NORM_MINMAX);
            // debug by printing the disparity map
            cout<<"Disparity map: "<<disparityMap<<endl;
            // return the disparity map
            cout<<"<<< computeDisparityMap() returned"<<endl;
            return disparityMap;
        }

        // function to test the implementation
        void sample(Mat img1, Mat img2) {

            Ptr<SIFT> detector = SIFT::create();

            std::vector<KeyPoint> keypoints1, keypoints2;
            Mat descriptors1, descriptors2;

            detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
            detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

            BFMatcher matcher(NORM_L2);
            std::vector<DMatch> matches;
            matcher.match(descriptors1, descriptors2, matches);

            Mat img_matches;
            drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);

            imshow("Matches", img_matches);
            cvFactory.saveImage("../output/CastleMatchesFromCV.jpg", img_matches);

            // Find the fundamental matrix and eliminate outliers using RANSAC
            std::vector<Point2f> points1, points2;
            for (const auto& match : matches) {
                points1.push_back(keypoints1[match.queryIdx].pt);
                points2.push_back(keypoints2[match.trainIdx].pt);
            }

            Mat inliers_mask;
            Mat F = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99, inliers_mask);

            std::vector<Point2f> inliers1, inliers2;
            for (int i = 0; i < inliers_mask.rows; ++i) {
                if (inliers_mask.at<uint8_t>(i)) {
                    inliers1.push_back(points1[i]);
                    inliers2.push_back(points2[i]);
                }
            }

            Mat img_inliers;
            drawMatches(img1, keypoints1, img2, keypoints2, matches, img_inliers, Scalar::all(-1), Scalar::all(-1), inliers_mask);
            imshow("Inliers", img_inliers);
            cvFactory.saveImage("../output/CastleInliersFromCV.jpg", img_inliers);

            // Compute the dense disparity map
            Mat gray1, gray2;
            cvtColor(img1, gray1, COLOR_BGR2GRAY);
            cvtColor(img2, gray2, COLOR_BGR2GRAY);

            Mat disp;
            Ptr<StereoBM> bm = StereoBM::create(64, 9);
            bm->compute(gray1, gray2, disp);

            // Display the disparity map
            Mat disp_show;
            normalize(disp, disp_show, 0, 255, NORM_MINMAX, CV_8U);
            imshow("Disparity", disp_show);
            cvFactory.saveImage("../output/CastleDisparityFromCV.jpg", disp_show);

            Mat disp_x;
            disp.convertTo(disp_x, CV_32F, 1.0 / 16.0); // scale factor for 16-bit disparity
            disp_x = disp_x.colRange(0, disp_x.cols / 2); // keep only x disparity
            imshow("Disparity in X direction", disp_x);
            cvFactory.saveImage("../output/CastleDisparityXFromCV.jpg", disp_x);

            Mat disp_y;
            disp.convertTo(disp_y, CV_32F, 1.0 / 16.0); // scale factor for 16-bit disparity
            disp_y = disp_y.colRange(disp_y.cols / 2, disp_y.cols); // keep only y disparity
            imshow("Disparity in Y direction", disp_y);
            cvFactory.saveImage("../output/CastleDisparityYFromCV.jpg", disp_y);

            // Convert disparity matrix to polar coordinates and create color image
            // Mat magnitude, angle;
            // cartToPolar(disp.colRange(0, disp.cols / 2), Mat::zeros(disp.size(), CV_32F), magnitude, angle);
            // Mat hsv;
            // hsv.create(disp.size(), CV_32FC3);
            // hsv.setTo(Scalar(0, 1, 0));
            // normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
            // hsv.col(0) = angle * 180 / M_PI / 2;
            // hsv.col(1) = magnitude;
            // Mat color_map;
            // applyColorMap(hsv, color_map, COLORMAP_HSV);

            // Display the disparity vector
            // imshow("Disparity Vector", color_map);

            waitKey();

        }
};
