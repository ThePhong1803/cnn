#include <iostream>
#include <vector>
#include <iomanip>
#include <time.h>
#include <neuralnetwork.h>
#include <assert.h>
#include <convolutionalLayer.h>
#include <eigen3/Eigen/Eigen>
#include <imagedata.h>
#include <math.h>

Scalar linear(Scalar x) { return x; }

int main() {
	srand(time(NULL));
	// prepare image data matrix
	ImageData img("test-img/test.bmp");
	Matrix mat(img.height, img.width);
	img.getPixelMatrix(&mat);
	std::vector<Matrix*> vec;
	vec.push_back(&mat);
	// config convolutional layer
	LayerConfig config;
	config.imageHeight  = img.height;
	config.imageWidth   = img.width;
	config.imageDepth   = 1;
	config.kernelHeight = 5;
	config.kernelWidth  = 5;
	config.numKernel    = 1;
	config.padding      = 0;
	config.striding     = 1;
	config.actFun		= linear;
	
	// create convolutional layer
	ConvolutionalLayer ConvLayer(&config);
	ConvLayer.propagateForward(vec);
	char map[] = " .:-=+*#%@";
	int idx = 0;
	for(int i = 0; i < ConvLayer.output[0] ->rows(); i++){
		for(int j = 0; j < ConvLayer.output[0] -> cols(); j++){
			idx = (int)ConvLayer.output[0] -> coeff(i, j) % 10;
			std::cout << map[idx];
		}
		std::cout << std::endl;
	}
	return 0;
}