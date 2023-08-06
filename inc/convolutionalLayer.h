#ifndef __CONVLAYER_H__
#define __CONVLAYER_H__

#include <interface.h>
#include <common.h>
#include <activation.h>

// method needed for this layer
Matrix corr(Matrix & mat, Matrix & kernel, uint32_t padding = 0, uint32_t striding = 1);
Matrix conv(Matrix & mat, Matrix & kernel, uint32_t padding = 0, uint32_t striding = 1);

class ConvConfig : public LayerConfig{
	public:
	ConvConfig();
	~ConvConfig();
	// image parameter
	uint32_t inputHeight;
	uint32_t inputWidth;
	uint32_t inputDepth;
	
	// kernel parameter
	uint32_t kernelHeight;
	uint32_t kernelWidth;
	uint32_t numKernel;
	
	// layer operation config
	uint32_t padding;
	uint32_t striding;
	
	// layer activation function config
	ScalarFunPtr actFun;
	ScalarFunPtr dactFun;

	// define layer learning rate
	Scalar learningRate;
	uint32_t &inputHeightRef();
	uint32_t &inputWidthRef();
	uint32_t &inputDepthRef();
	
	uint32_t &kernelHeightRef();
	uint32_t &kernelWidthRef();
	uint32_t &kernelDepthRef();

	ScalarFunPtr &activationFunctionRef() override;
	ScalarFunPtr &activationFunctionDerivativeRef() override;
};

class ConvolutionalLayer : public Layer {
    public:
	ConvConfig* 						config;			// define layer topology, typically about the size of input, size and number of kernel
    std::vector<Matrix*> 				output;			// output of convolutional layer
	std::vector<Matrix*> 				caches;			// store the value before activation
	std::vector<std::vector<Matrix*>> 	kernel;			// kernel matrix container
    std::vector<Matrix*> 				biases;			// bias for the layer
	std::vector<Matrix*> 				input;			// input of convolutional layer
	
	// layer constructor and destructor
    ConvolutionalLayer(ConvConfig *_config);
    ~ConvolutionalLayer();
	
	// attribute access method
	std::vector<Matrix *> &inputRef() override;
	std::vector<Matrix *> &outputRef() override;

    // this function perform network forward propagation
	void propagateForward(std::vector<Matrix*> * input) override;
	
	// thsi frunction perform network backward propagateion
	void propagateBackward(std::vector<Matrix*> * errors) override;
};
#endif