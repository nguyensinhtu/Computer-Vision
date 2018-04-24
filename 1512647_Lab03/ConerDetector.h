#pragma 
#define _USE_MATH_DEFINES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <math.h>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

class ConerDetector
{
public:
	
	// Gaussian Filter Generation
	double FilterWindow(int kernel[][5], double alpha); 

	double FilterWindow_LoG(int kernel[][5], double sigma);

	cv::Mat GaussianFilter(cv::Mat &srcImage, double sigma);

	cv::Mat LoG(cv::Mat &srcImge, double sigma);

	cv::Mat detectHarris(cv::Mat &srcImg);

	cv::Mat Blob(cv::Mat &srcImg);

	cv::Mat DoG(cv::Mat &srcImg);

	double Derivative(uchar *pdata, int kernel[3][3], int width, int height, int y, int x, int widthstep, int nchanel, int ii);

	ConerDetector();
	~ConerDetector();
};

