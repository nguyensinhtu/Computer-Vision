#include "ConerDetector.h"


int main(int argc, char **argv) {
	Mat srcimg = imread(argv[1], CV_LOAD_IMAGE_ANYCOLOR);
	if (!srcimg.data) {
		cout << "khong load duoc anh";
		return 0;
	}

	ConerDetector *conerdetector = new ConerDetector();

	//conerdetector->gaussianfilter(srcimg, 1);

	conerdetector->detectHarris(srcimg);
	//conerdetector->Blob(srcimg);
	/*double sigma = 1.4;
	double kernel[9][9];

	double tmp[9][9];
	double r, s = 2 * sigma*sigma, ss = M_PI*sigma*sigma*sigma*sigma;

	double mmin = 1000;
	for (int x = -4; x <= 4; ++x) {
		for (int y = -4; y <= 4; ++y) {
			tmp[x + 4][y + 4] = (1.0 / ss)*(1 - ((x*x + y*y) / s))*exp(-(x*x + y*y) / s);
			mmin = min(mmin, tmp[x + 4][y + 4]);
		}
	}

	cout << mmin << endl;
	for (int i = 0; i < 9; ++i)
		for (int j = 0; j < 9; ++j) {
			kernel[i][j] = (int)round(tmp[i][j] / mmin);
		}

	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j)
			cout << kernel[i][j] << " ";
		cout << endl;
	}*/
	return 0;
}

