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

	//template<typename T>
	cv::Mat GaussianFilter(cv::Mat &srcImage, double sigma, int type);

	cv::Mat LoG(cv::Mat &srcImge, vector<Mat> &ScaleSpace);

	cv::Mat detectHarris(cv::Mat &srcImg);

	cv::Mat Blob(cv::Mat &srcImg, int type);

	// hàm xấp xỉ Laplace of Gaussian
	// input : hai ảnh xám
	//		 : sigma1 cho ảnh 1, sigma2 cho ảnh 2
	// ouput : ảnh đã xấp xỉ Laplace of Gaussian
	void DoG(cv::Mat &srcImg, vector<Mat> &ScaleSpace);


	// hàm tính đạo hàm của ảnh tai 1 điểm
	// input : 
	// @pram1 : 
	double Derivative(uchar *pdata, int kernel[3][3], int width, int height, int y, int x, int widthstep, int nchanel, int ii);

	float mul(int y, int x, int kernel[][5], float *pdata, int width, int height, int widthstep, int nchanel, int ii);

	float mul1(int y, int x, int kernel[][5], uchar *pdata, int width, int height, int widthstep, int nchanel, int ii);

	float mul(int y, int x, int kernel[][5], uchar *pdata, int width, int height);

	ConerDetector();
	~ConerDetector();
};





// point và radius
struct BlobPoint {
	Point p;
	double radius;
};
