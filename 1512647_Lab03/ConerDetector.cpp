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

// khử nhiễu trên ảnh xám
// input : ảnh xám

Mat ConerDetector::GaussianFilter(Mat & srcImage, double sigma, int type)
{
	int nrows = srcImage.rows;
	int ncols = srcImage.cols;

	int kernel[5][5] = { { 0 } };

	double w = FilterWindow(kernel, sigma);

	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < 5; ++j)
			std::cout << kernel[i][j] << " ";
		std::cout << endl;
	}

	Mat dstImg(nrows, ncols, srcImage.type());

	uchar *data1 = NULL; float *data2 = NULL;
	if (type == 1)
		data1 = (uchar*)srcImage.data;
	else if (type == 2)
		data2 = (float*)srcImage.data;

	for (int y = 0; y < nrows; ++y) {
		float *pRow = dstImg.ptr<float>(y);
		for (int x = 0; x < ncols; x++) {
			if (type == 1)
				pRow[x] = (float)(w * mul1(y, x, kernel, data1, ncols, nrows, ncols, 1, 0));
			else if (type == 2)
				pRow[x] = (float)(w * mul(y, x, kernel, data2, ncols, nrows, ncols, 1, 0));
		}
	}

	return dstImg;
}

// nhân window với kernel
float ConerDetector::mul(int y, int x, int kernel[][5], float *pdata, int width, int height, int widthstep, int nchanel, int ii) {
	float res = 0.0;
	//float * data = (float*)pdata;
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

float ConerDetector::mul1(int y, int x, int kernel[][5], uchar *pdata, int width, int height, int widthstep, int nchanel, int ii) {
	float res = 0.0;
	
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

float ConerDetector::mul(int y, int x, int kernel[][5], uchar *pdata, int width, int height) {
	float res = 0.0;
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
	double mmin = 10000.0;
	double tmp[5][5] = { {0.0} };
	for (int x = -2; x <= 2; ++x) {
		for (int y = -2; y <= 2; ++y) {
			r = (x*x + y*y)*1.0;
			tmp[x + 2][y + 2] = 
				(exp(-(r / s)) / (M_PI * s));
			mmin = min(tmp[x + 2][y + 2], mmin);
		}
	}
	cout << "mmin : " << mmin << endl;
	for (int x = -2; x <= 2; ++x)
		for (int y = -2; y <= 2; ++y) {
			kernel[x + 2][y + 2] = round(tmp[x + 2][y + 2] / mmin);
		}

	return (mmin);
}

// tạo windows filter for laplace
double ConerDetector::FilterWindow_LoG(int kernel[][5], double sigma)
{
	double tmp[5][5] = { {0.0} };
	double r = 0.0, s = 2.0 * sigma*sigma, ss = 3.14159265359*sigma*sigma*sigma*sigma;

	double mmin = 1e9 + 10.0;
	for (int x = -2; x <= 2; ++x) {
		for (int y = -2; y <= 2; ++y) {
			r = (double)(x*x + y*y);
			double k = (double)(r / s);
			if (fabs(r - s) < 0.0000000001*fabs(r)) {
				k = 0;
			}
			else
				k = k - 1.0;

			tmp[x + 2][y + 2] = (1.0 / ss)*k*exp(-r / s);
			if (tmp[x + 2][y + 2] != 0.0)
				mmin = min(mmin, abs(tmp[x + 2][y + 2]));
		}
	}

	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 5; ++j) {
			kernel[i][j] = (int)round(tmp[i][j]/(mmin));
		}

	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < 5; ++j) {
			std::cout << kernel[i][j] << " ";
		}
		std::cout << endl;
	}

	cout << "mmin " << mmin << " sigma : " << sigma << endl;

	return mmin;
}




cv::Mat ConerDetector::LoG(cv::Mat & srcImage, vector<Mat> &ScaleSpace)
{
	int nrows = srcImage.rows;
	int ncols = srcImage.cols;
	double sigma = 1.0;
	int n = 3;

	int kernel[5][5] = { { 0 } };

	double w;

	Mat dstImg(nrows, ncols, CV_8UC1);

	uchar *data = (uchar*)srcImage.data;

	for (int i = 0; i < n; ++i) {
		w = FilterWindow_LoG(kernel, sigma);
		for (int y = 0; y < nrows; ++y) {
			uchar *pRow = dstImg.ptr<uchar>(y);
			for (int x = 0; x < ncols; x++) {
				pRow[x] = (uchar)round((pow(sigma, 2) * w * mul1(y, x, kernel, data, ncols, nrows, ncols, 1, 0)));
			}
		}

		sigma *= sqrt(2);
		ScaleSpace.push_back(dstImg);
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
	Mat IIxx = GaussianFilter(Ixx, 1.0, 2);
	Mat IIyy = GaussianFilter(Iyy, 1.0, 2);
	Mat IIxy = GaussianFilter(Ixy, 1.0, 2);



	vector<Point> points;

	Mat rr(height, width, CV_32FC1);

	for (int y = 0; y < height; ++y) {
		float *prow1 = IIxx.ptr<float>(y);
		float *prow2 = IIyy.ptr<float>(y);
		float *prow3 = IIxy.ptr<float>(y);
		float *prow4 = rr.ptr<float>(y);
		for (int x = 0; x < width; ++x) {
			prow4[x] = (prow1[x] * prow2[x] - prow3[x] * prow3[x]) - 0.04*(prow1[x] + prow2[x]);
		}
	}


	float *rrdata = (float*)rr.data;
	
	for (int y = 0; y < height; ++y) {
		float *prow = rr.ptr<float>(y);
		for (int x = 0; x < width; ++x) {
			float R = prow[x];

			if (R > 0) {
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
		
		// đánh dấu những điểm góc cạnh
		drawMarker(srcImg, points[i], Scalar(0, 0, 255));
	}

	namedWindow("Coner detection", WINDOW_AUTOSIZE);
	cv::imshow("Coner detection", srcImg);
	cv::waitKey(0);
	return srcImg;
}

cv::Mat ConerDetector::Blob(cv::Mat & srcImg, int type)
{
	Mat dst;
	if (!isGrayScale(srcImg))
		RGB2GrayScale(srcImg, dst);
	else {
		dst = srcImg.clone();
	}

	int width = dst.cols;
	int height = dst.rows;

	vector<Mat> ScaleSpace;
	if (type == 1)
		LoG(dst, ScaleSpace);
	else if (type == 2)
		DoG(dst, ScaleSpace);

	vector<Point> points;
	cout << "scale space : " << ScaleSpace.size() << endl;
	int sz = ScaleSpace.size();
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			bool flag = true;
			for (int ii = 1; ii < sz - 1; ++ii) {
				uchar *data1 = (uchar*)ScaleSpace[ii - 1].data;
				uchar *data2 = (uchar*)ScaleSpace[ii].data;
				uchar *data3 = (uchar*)ScaleSpace[ii + 1].data;
				uchar *prow1 = ScaleSpace[ii].ptr<uchar>(y);
				int stepwidth = ScaleSpace[ii].step[0];
				int r = prow1[x];
				for (int i = -2; i <= 2; ++i) {
					for (int j = -2; j <= 2; ++j) {
						if (y + i >= height || y + i < 0 || x + j >= width || x + j < 0)
							continue;
						int a = data1[(y + i)*stepwidth + (x + j)];
						int b = data2[(y + i)*stepwidth + (x + j)];
						int c = data3[(y + i)*stepwidth + (x + j)];

						//cout << r << " " << a << " " << b << " " << c << endl;
						if (r < a || r < b || r < c) {
							//cout << r << " " << a << " " << b << " " << c << endl;
							flag = false;
							break;
						}
					}

					if (!flag)
						break;
				}
				
				if (flag)
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

			/*data[(points[i].y)*widthstep + (points[i].x)*nchanel] = 0;
			data[(points[i].y)*widthstep + (points[i].x)*nchanel + 1] = 0;
			data[(points[i].y)*widthstep + (points[i].x)*nchanel + 2] = 255;*/
			circle(srcImg, points[i], 10, Scalar(0, 0, 255));
		}

		namedWindow("Filter", WINDOW_AUTOSIZE);
		cv::imshow("Filter", srcImg);
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


void ConerDetector::DoG(Mat & srcImg, vector<Mat> &ScaleSpace)
{
	int nrows = srcImg.rows;
	int ncols = srcImg.cols;
	double sigma = 1.0;
	int n = 7;

	Mat *Img1 = new Mat(nrows, ncols, srcImg.type());
	Mat *Img2 = new Mat(nrows, ncols, srcImg.type());
	Mat dst(nrows, ncols, CV_8UC1);
	*Img1 = GaussianFilter(srcImg, sigma, 1); sigma *= sqrt(2);

	uchar *data = (uchar*)srcImg.data;

	for (int i = 0; i < n; ++i) {
		*Img2 = GaussianFilter(srcImg, sigma, 1);

		for (int y = 0; y < nrows; ++y) {
			uchar *prow1 = Img1->ptr<uchar>(y);
			uchar *prow2 = Img2->ptr<uchar>(y);
			uchar *pdst = dst.ptr<uchar>(y);

			for (int x = 0; x < ncols; ++x) {
				pdst[x] = (uchar)(prow2[x] - prow1[x]);
			}
		}
		sigma *= sqrt(2);
		Img1->release();
		Img1 = Img2;
		ScaleSpace.push_back(dst);
	}

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

	return (double)(res*(1.0/4)*(1.0/255));
}

ConerDetector::ConerDetector()
{
}


ConerDetector::~ConerDetector()
{
}
