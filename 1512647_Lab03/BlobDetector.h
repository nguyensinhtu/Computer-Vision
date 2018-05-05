#pragma once
#ifndef _BLOB_DETECTOR_

#define _BLOB_DETECTOR_

#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <cstdio>
using namespace cv;
using namespace std;
#define RED_CHANNEL 2
#define GREEN_CHANNEL 3
#define BLUE_CHANNEL 4


class BlobDetector {
public:
	void FilterWindow_LoG(double kernel[][5], double sigma);
	Mat Convolution(const Mat& srcImg, double sigma);
	Mat test(Mat srcImg);
	float CalRadius(int k);
	BlobDetector(const Mat& srcImg);
	BlobDetector();
	~BlobDetector();


};


#endif