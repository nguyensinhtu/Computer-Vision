#include "ConerDetector.h"

// kiểm tra một ảnh có phải là ảnh xám
// input : ảnh cần kiểm tra
// output : trả về true nếu là ảnh xám
//			trả về false ngược lại
bool isGrayScale(const cv::Mat &image) {
	if (image.type() == CV_8UC1)
		return true;
	return false;
}

const float cr = 0.299, cg = 0.587, cb = 0.114;
// công thức chuyển một rgb -> gray
// input : giá trị r, g, b của pixel
// ouput : giá trị xám
int formular1(int r, int g, int b) {
	float fr = r / 255.0;
	float fg = g / 255.0;
	float fb = b / 255.0;
	return (int)((cr  * fr + cg * fg + cb * fb)*255.0);
}

// Hàm chuyển đổi ảnh màu sang ảnh xám 
// input : ảnh màu
// ouput : ảnh xám
int RGB2GrayScale(const cv::Mat& sourceImage, cv::Mat& destinationImage) {
	if (!sourceImage.data) {
		return 0;
	}

	int width = sourceImage.cols;
	int height = sourceImage.rows;
	destinationImage.create(height, width, CV_8UC1);

	for (int y = 0; y < height; ++y) {
		const unsigned char* data = sourceImage.ptr<uchar>(y);
		unsigned char *data1 = destinationImage.ptr<uchar>(y);
		for (int x = 0; x < width; ++x) {
			*data1 = formular1(*data++, *data++, *data++); data1++;
		}
	}

	return 1;
}

// nhân window với kernel
double mul(int y, int x, int kernel[][5], float *pdata, int width, int height, int widthstep, int nchanel, int ii) {
	int res = 0.0;
	for (int i = -2; i <= 2; ++i) {
		for (int j = -2; j <= 2; ++j) {
			if (y + i >= height || y + i < 0 || x + j >= width || x + j < 0) {
				continue;
			}
			else {
				//cout << (double)pdata[(y + i)*widthstep + (x + j)*nchanel + ii] << endl;
				res += pdata[(y + i)*widthstep + (x + j)*nchanel + ii] * kernel[2 - i][2 - j];
			}
		}
	}
	return res;
}

double mul(int y, int x, int kernel[][5], uchar *pdata, int width, int height) {
	int res = 0.0;
	for (int i = -2; i <= 2; ++i) {
		for (int j = -2; j <= 2; ++j) {
			if (y + i >= height || y + i < 0 || x + j >= width || x + j < 0) {
				continue;
			}
			else {
				//cout << (double)pdata[(y + i)*widthstep + (x + j)*nchanel + ii] << endl;
				res += pdata[(y + i) * width + (x + j)*1 + 0] * kernel[2 - i][2 - j];
			}
		}
	}
	return res;
}
// tạo kernel khử nhiễu
double ConerDetector::FilterWindow(int kernel[][5], double sigma)
{
	double r, s = 2.0 * sigma * sigma;

	for (int x = -2; x <= 2; ++x) {
		for (int y = -2; y <= 2; ++y) {
			r = sqrt(x*x + y*y);
			kernel[x + 2][y + 2] = 
				(int)round(((exp(-(r*r) / s)) / (M_PI * s)) / (1.0/273));
		}
	}
	
	return (1.0 / 273);
}

// tạo windows filter for laplace
double ConerDetector::FilterWindow_LoG(int kernel[][5], double sigma)
{
	double tmp[5][5];
	double r, s = 2 * sigma*sigma, ss = M_PI*sigma*sigma*sigma*sigma;

	double mmin = 1000;
	for (int x = -2; x <= 2; ++x) {
		for (int y = -2; y <= 2; ++y) {
			tmp[x + 2][y + 2] = (1.0 / ss)*(1 - ((x*x + y*y) / s))*exp(-(x*x + y*y) / s);
			mmin = min(mmin, tmp[x + 2][y + 2]);
		}
	}

	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 5; ++j) {
			kernel[i][j] = (int)round(tmp[i][j]/mmin);
		}

	return 1;
}


// khử nhiễu trên 
Mat ConerDetector::GaussianFilter(Mat & srcImage, double alpha)
{
	int nrows = srcImage.rows;
	int ncols = srcImage.cols;

	int kernel[5][5] = { {0} };

	double w = FilterWindow(kernel, 1.0);

	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < 5; ++j)
			std::cout << kernel[i][j] << " ";
		std::cout << endl;
	}

	Mat dstImg(nrows, ncols, CV_32FC1);

	float *data = (float*)srcImage.data;
	for (int y = 0; y < nrows; ++y) {
		float *pRow = srcImage.ptr<float>(y);
		for (int x = 0; x < ncols; x++) {
			pRow[x] = (double)(w * mul(y, x, kernel, data, ncols, nrows, ncols, 1, 0));
		}
	}

	return dstImg;
}

cv::Mat ConerDetector::LoG(cv::Mat & srcImage, double sigma)
{
	int nrows = srcImage.rows;
	int ncols = srcImage.cols;

	int kernel[5][5] = { { 0 } };

	double w = FilterWindow_LoG(kernel, sigma);

	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < 5; ++j) {
			std::cout << kernel[i][j] << " ";
		}
		std::cout << endl;
	}

	Mat dstImg(nrows, ncols, CV_8UC1);

	uchar *data = (uchar*)srcImage.data;

	for (int y = 0; y < nrows; ++y) {
		uchar *pRow = dstImg.ptr<uchar>(y);
			for (int x = 0; x < ncols; x++) {
				pRow[x] = (uchar)(w * mul(y, x, kernel, data, ncols, nrows));
		}
	}
	return dstImg;
}

Mat ConerDetector::detectHarris(Mat & srcImg)
{
	Mat dst;
	if (!isGrayScale(srcImg))
		RGB2GrayScale(srcImg, dst);
	else {
		dst = srcImg.clone();
	}

	int width = dst.cols;
	int height = dst.rows;

	Mat Ixx(height, width, CV_32FC1);
	Mat Iyy(height, width, CV_32FC1);
	Mat Ixy(height, width, CV_32FC1);

	// kerneals
	int Wx[3][3] = { { 1, 0, -1 },
						{ 2, 0, -2 },
						{ 1, 0, -1 } };

	int Wy[3][3] = { { -1, -2, -1 },
						{ 0, 0, 0 },
						{ 1, 2, 1 } };
	
	Mat newdst(dst.clone());
	for (int y = 0; y < height; ++y) {

		float *prow2 = Ixx.ptr<float>(y);
		float *prow3 = Iyy.ptr<float>(y);
		float *prow4 = Ixy.ptr<float>(y);

		uchar *pdata = newdst.ptr<uchar>(y);
		for (int x = 0; x < width; ++x) {
			double ix = Derivative(dst.data, Wx, width, height, y, x, width, 1, 0);
			double iy = Derivative(dst.data, Wy, width, height, y, x, width, 1, 0);
			
			pdata[x] = (uchar)(sqrt(ix*ix + iy*iy));
			prow2[x] = ix*ix;
			prow3[x] = iy*iy;
			prow4[x] = ix*iy;

		}
	}
	
	// áp bộ lọc gaussian lên các ma trận Ixx, Iyy, Ixy;
	Mat IIxx = GaussianFilter(Ixx, 1);
	Mat IIyy = GaussianFilter(Iyy, 1);
	Mat IIxy = GaussianFilter(Ixy, 1);

	//Ixx.release(); Iyy.release(); Ixy.release();

	/*cout << dst.size() << endl;
	cout << Ixx.size() << endl;
	cout << Iyy.size() << endl;
	cout << Ixy.size() << endl;*/


	vector<Point> points;

	Mat rr(height, width, CV_32FC1);

	for (int y = 0; y < height; ++y) {
		float *prow1 = Ixx.ptr<float>(y);
		float *prow2 = Iyy.ptr<float>(y);
		float *prow3 = Ixy.ptr<float>(y);
		float *prow4 = rr.ptr<float>(y);
		for (int x = 0; x < width; ++x) {
			prow4[x] = (prow1[x] * prow2[x] - prow3[x] * prow3[x]) - 0.06*(prow1[x] + prow2[x]);
		}
	}


	float *rrdata = (float*)rr.data;
	
	for (int y = 0; y < height; ++y) {
		float *prow = rr.ptr<float>(y);
		for (int x = 0; x < width; ++x) {
			float R = prow[x];

			if (R > 10000000) {
				bool flag = true;
				for (int i = -1; i <= 1; ++i) {
					for (int j = -1; j <= 1; ++j) {
						if (y + i >= height || y + i < 0 || x + j >= width && x + j < 0)
							continue;
						if (R < rrdata[(y + i)*width + (x + j)]) {
							flag = false;
							break;
						}
					}

					if (!flag)
						break;
				}

				if (flag)
					points.push_back(Point(x, y));

			}
		}
	}


	// xac dinh corner ten anh goc;
	int widthstep = srcImg.step[0];
	int nchanel = srcImg.step[1];
	uchar *data = srcImg.data;
	std::cout << "points : " << points.size();
	for (int i = 0; i < points.size(); ++i) {
		/*for (int l = -2; l <= 2; ++l)
			for (int k = -2; k <= 2; ++k) {
				if (points[i].y + l >= height || points[i].y + l < 0 || points[i].x + k >-width || points[i].x + k < 0)
					continue;
				
				data[(points[i].y + l)*widthstep + (points[i].x + k)*nchanel] = 0;
				data[(points[i].y + l)*widthstep + (points[i].x + k)*nchanel + 1] = 0;
				data[(points[i].y + l)*widthstep + (points[i].x + k)*nchanel + 2] = 255;
			}*/

		data[(points[i].y)*widthstep + (points[i].x)*nchanel] = 0;
		data[(points[i].y )*widthstep + (points[i].x)*nchanel + 1] = 0;
		data[(points[i].y)*widthstep + (points[i].x)*nchanel + 2] = 255;
	}

	namedWindow("Filter", WINDOW_AUTOSIZE);
	cv::imshow("Filter", srcImg);
	//imshow("Filter", newdst);
	cv::waitKey(0);
	return srcImg;
}

cv::Mat ConerDetector::Blob(cv::Mat & srcImg)
{
	Mat dst;
	if (!isGrayScale(srcImg))
		RGB2GrayScale(srcImg, dst);
	else {
		dst = srcImg.clone();
	}

	int width = dst.cols;
	int height = dst.rows;

	Mat dst1 = LoG(dst, 1);
	Mat dst2 = LoG(dst, sqrt(2));
	Mat dst3 = LoG(dst, 2.0*sqrt(2));

	vector<Point> points;
	uchar *data1 = (uchar*)dst1.data;
	uchar *data2 = (uchar*)dst2.data;
	uchar *data3 = (uchar*)dst3.data;

	int stepwidth = dst1.step[0];
	
	for (int y = 0; y < height; ++y) {
		
		uchar *prow1 = dst2.ptr<uchar>(y);
		for (int x = 0; x < width; ++x) {
			bool flag = true;
			int r = prow1[x];
			for (int i = -1; i <= 1; ++i) {
				for (int j = -1; j <= 1; ++j) {
					if (y + i >= height || y + i < 0 || x + j >= width || x + j < 0)
						continue;
					int a = data1[(y + i)*stepwidth + (x + j)];
					int b = data2[(y + i)*stepwidth + (x + j)];
					int c = data3[(y + i)*stepwidth + (x + j)];

					if (r < a || r < b || r < c) {
						flag = false;
						break;
					}
				}

				if (!flag)
					break;
			}
			if (flag)
				points.push_back(Point(x, y));
		}
	}

		// xac dinh corner ten anh goc;
		int widthstep = srcImg.step[0];
		int nchanel = srcImg.step[1];
		uchar *data = srcImg.data;
		std::cout << "points : " << points.size();
		for (int i = 0; i < points.size(); ++i) {

			data[(points[i].y)*widthstep + (points[i].x)*nchanel] = 0;
			data[(points[i].y)*widthstep + (points[i].x)*nchanel + 1] = 0;
			data[(points[i].y)*widthstep + (points[i].x)*nchanel + 2] = 255;
		}

		namedWindow("Filter", WINDOW_AUTOSIZE);
		cv::imshow("Filter", dst1);
		//imshow("Filter", newdst);
		cv::waitKey(0);
	return srcImg;
}

bool checkpoint(uchar *data, int x, int y, int height, int width, int widthstep) {
	// check (y, x) với những điểm xung quanh
	for (int i = -1; i <= 1; ++i) {
		for (int j = -1; j <= 1; ++j) {
			if (y + i >= width || y + i < 0 || x + j >= width || x + j < 0)
				continue;
			if (data[y*widthstep + x] < data[(y + i)*widthstep + (x + j)])
				return false;
		}
	}
}

Mat ConerDetector::DoG(Mat & srcImg)
{
	Mat dst;
	RGB2GrayScale(srcImg, dst);
	double alpha = 1.0;
	vector<Mat> Scales;
	Mat dst1, dst2;
	double sqrt2 = sqrt(2);
	dst1 = GaussianFilter(dst, alpha);
	for (int i = 0; i < 6; ++i) {
		dst2 = GaussianFilter(dst, alpha*sqrt2);
		dst1 = dst2;
		alpha *= sqrt2;
		Scales.push_back(dst2 - dst1);
	}

	//
	return srcImg;
}

double ConerDetector::Derivative(uchar * pdata, int kernel[3][3], int width, int height, int y, int x, int widthstep, int nchanel, int ii)
{
	double res = 0.0;
	for (int i = -1; i <= 1; ++i) {
		for (int j = -1; j <= 1; ++j) {
			if (y + i >= height || y + i < 0 || x + j >= width || x + j < 0)
				continue;
			else {
				res += pdata[(y + i)*widthstep + (x + j)*nchanel + ii] * kernel[1 - i][1 - j];
			}
		}
	}

	return (double)res*(1.0/4);
}

ConerDetector::ConerDetector()
{
}


ConerDetector::~ConerDetector()
{
}
