#include "ConerDetector.h"


int main(int argc, char **argv) {
	Mat srcimg = imread(argv[1], CV_LOAD_IMAGE_ANYCOLOR);
	if (!srcimg.data) {
		cout << "khong load duoc anh";
		return 0;
	}

	ConerDetector *conerdetector = new ConerDetector();

	//conerdetector->detectHarris(srcimg);
	conerdetector->Blob(srcimg, 1);
	
	return 0;
}

