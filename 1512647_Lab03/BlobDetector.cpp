#include "stdafx.h"
#include "BlobDetector.h"

void BlobDetector::FilterWindow_LoG(double kernel[][5], double sigma)
{
	float s2 = sigma * sigma;

	for (int x = -2; x <= 2; ++x)
	{
		for (int y = -2; y <= 2; ++y)
		{
			double sumSquare = x * x + y * y;
			kernel[x + 2][y + 2] = ((sumSquare - 2 * s2)/(s2*s2))*(double)exp(-(sumSquare) / (2 * s2));
		}
	}
}

Mat BlobDetector::Convolution(const Mat & srcImg, double sigma)
{
	
	double filter[5][5];
	FilterWindow_LoG(filter, sigma);
	cout << "====================================\n";
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++)
		{
			cout << filter[i][j] << " ";
		} 
		cout << endl;
	}

	int Row = srcImg.rows;
	int Col = srcImg.cols;
	Mat dstImg = Mat(Row, Col, CV_32FC1);

	int src_stepWidth = srcImg.step[0];
	int src_nchannels = srcImg.step[1];

	uchar* src_data = (uchar*)srcImg.data;
	float * dst_data = (float*)dstImg.data;
	for (int y = 0; y < Row; ++y, dst_data += Col) {
		float *pRow = dst_data;
		for (int x = 0; x < Col; ++x, pRow += 1) {
			int yy, xx;
			float res = 0;
			for (int i = -2; i <= 2; i++) {
				for (int j = -2; j <= 2; j++) {
					yy = y + i;
					xx = x + j;
					if (yy >= 0 && yy < Row && xx >= 0 && xx < Col) {
						res += (float)src_data[yy*src_stepWidth + xx * src_nchannels] * filter[2-i][2-j];
					}
				}
			}
			res /= 255.0;
			res *= res;
			pRow[0] = res;
		}
	}
	return dstImg;
}

Mat BlobDetector::test( Mat srcImg)
{
	int cnt = 0 ;
	namedWindow("Display", WINDOW_AUTOSIZE);
	Mat temp = srcImg.clone();
	//imwrite("Blob.png", temp);
	//imshow("Display", temp);
	//waitKey(0);
	Mat dstImg;
	cv::cvtColor(srcImg, dstImg, cv::COLOR_RGB2GRAY);

	if (dstImg.type() == CV_8UC1) {
		cout << " TRue" << endl;
	}
	//imshow("Display", dstImg);
	///waitKey(0);
	Mat scale[8];
	scale[0] = Convolution(dstImg, 1);
	scale[1] = Convolution(dstImg, sqrt(2));
	scale[2] = Convolution(dstImg, 2);
	scale[3] = Convolution(dstImg, 2 * sqrt(2));
	scale[4] = Convolution(dstImg, 4);
	scale[5] = Convolution(dstImg, 4 * sqrt(2));
	scale[6] = Convolution(dstImg, 8);
	scale[7] = Convolution(dstImg, 8 * sqrt(2));

	//
	//Mat scale[8];
	//scale[0] = Convolution(dstImg, -1.2);
	//scale[1] = Convolution(dstImg, -1);
	//scale[2] = Convolution(dstImg, -0.8);
	//scale[3] = Convolution(dstImg, -0.6);
	//scale[4] = Convolution(dstImg, -0.4);
	//scale[5] = Convolution(dstImg, -0.2);
	//scale[6] = Convolution(dstImg, 0);
	//scale[7] = Convolution(dstImg, 0.2);

	/*
	Mat scale[5];
	scale[0] = Convolution(dstImg, 1);
	scale[1] = Convolution(dstImg, 2);
	scale[2] = Convolution(dstImg, 4);
	scale[3] = Convolution(dstImg, 8);
	scale[4] = Convolution(dstImg, 16);
	*/

	int Row = srcImg.rows;
	int Col = srcImg.cols;


	//float * dst_data[8];
	//for (int i = 0; i < 8; i++) {
	//	dst_data[i] = (float*)scale[i].data;
	//}


	//Duyệt ảnh để xác định các điểm ứng viên và xác định điểm đặc trưng
	for (int i = 0; i < Row; ++i) {	//Duyệt hàng
		for (int j = 0; j < Col; ++j) {	//Duyệt cột
			for (int k = 1; k <= 6; ++k) {
				float r;
				bool check = true;
				//Kiểm tra xem điểm ảnh có lớn hơn 8 điểm lận không
				for (int i2 = i - 1; i2 <= i + 1; ++i2) {
					for (int j2 = j - 1; j2 <= j + 1; ++j2) {
						//Xử lý truy cập các điểm ngoài ảnh
						if (i2 < 0 || j2 < 0 || i2 >= Row || j2 >= Col) continue;
						//Loại trừ điểm đang xét
						if (i2 == i && j2 == j) continue;
						//So sánh
						check &= (scale[k].at<float>(i, j) > scale[k].at<float>(i2, j2));
						if (!check)
							break;
					}
				}
				if (check) {
					//Kiểm tra xem điểm ảnh có lớn hơn 9 điểm tương ứng ở tầng trên và tầng dưới không không
					for (int i2 = i - 1; i2 <= i + 1; ++i2) {
						for (int j2 = j - 1; j2 <= j + 1; ++j2) {
							//Xử lý truy cập các điểm ngoài ảnh
							if (i2 < 0 || j2 < 0 || i2 >= srcImg.rows || j2 >= srcImg.cols) continue;
							//So sánh với tầng trên
							check &= (scale[k].at<float>(i, j) > scale[k + 1].at<float>(i2, j2));
							//So sánh với tầng dưới
							check &= (scale[k].at<float>(i, j) > scale[k - 1].at<float>(i2, j2));
							if (!check)
								break;
						}
					}
				}
				//Đánh dấu điểm đặc trưng
				if (check) {
					cnt++;
					circle(temp, Point(j,i), CalRadius(k), Scalar(0, 0, 255));
					cout << k << endl;
				}
			}
		}
	}
	cout << "cnt = " << cnt << endl;
	imwrite("Blob.png", temp);
	imshow("Display", temp);
	waitKey(0);
	return temp;
}

float BlobDetector::CalRadius(int k)
{
	float res = 0;
	switch (k) {
	case 1:
		res = 2;
		break;
	case 2:
		res = 2 * sqrt(2);
		break;
	case 3:
		res = 4;
		break;
	case 4:
		res = 4 * sqrt(2);
		break;
	case 5:
		res = 8;
		break;
	case 6:
		res = 8 * sqrt(2);
		break;
	default :
		break;

	}
	return res*res;
	

	//switch (k) {
	//case 1:
	//	res = -1;
	//	break;
	//case 2:
	//	res = -0.8;
	//	break;
	//case 3:
	//	res = -0.6;
	//	break;
	//case 4:
	//	res = -0.4;
	//	break;
	//case 5:
	//	res = -0.2;
	//	break;
	//case 6:
	//	res = 0;
	//	break;
	//default:
	//	break;

	//}
	//res *= sqrt(2);
	//res *= res;
	//return res + 1;
	
	/*
	switch (k) {
	case 1:
		res = 2*sqrt(2);
		break;
	case 2:
		res = 4*sqrt(2);
		break;
	case 3:
		res = 8*sqrt(2);
		break;
	default:
		break;
	}
	return res;
	*/
	return res;
}

BlobDetector::BlobDetector(const Mat & srcImg)
{
}

BlobDetector::BlobDetector()
{
}

BlobDetector::~BlobDetector()
{
}
