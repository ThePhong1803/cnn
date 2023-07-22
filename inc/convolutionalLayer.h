#ifndef __CONVLAYER_H__
#define __CONVLAYER_H__

#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>
#include <assert.h>
#include <interface.h>

// function perform correlation and convolution operaton on matrices
Matrix corr(Matrix &mat, Matrix &kernel, uint32_t padding = 1, uint32_t striding = 1);
Matrix conv(Matrix &mat, Matrix &kernel, uint32_t padding = 1, uint32_t striding = 1);

struct LayerConfig {
	// image parameter
	int imageHeight;
	int imageWidth;
	int imageDepth;
	
	// kernel parameter
	int kernelHeight;
	int kernelWidth;
	int kernelDepth;
	int numKernel;
	
	// layer operation config
	int padding;
	int striding;
};

class ConvolutionalLayer : public Layer {
    public:
	LayerConfig* 						config;			// define layer topology, typically about the size of input, size and number of kernel
    std::vector<Matrix*> 				output;			// output of convolutional layer
	std::vector<std::vector<Matrix*>> 	kernel;			// kernel matrix container
    std::vector<Matrix*> 				bias;			// bias for the layer
	
	// layer constructor and destructor
    ConvolutionalLayer(LayerConfig * _config);
    ~ConvolutionalLayer();
	
	// this function perform network forward propagation
	void propagateForward(std::vector<Matrix> &input);
	
	// thsi frunction perform network backward propagateion
	void propagateBackward(std::vector<Matrix> &output);
};
#endif