#ifndef __LAYER_H__
#define __LAYER_H__
#include <common.h>

/* Class layer definition prototype*/
class Layer;
class ConvolutionalLayer;
class DenseLayer;
class FlattenLayer;
class MaxPoolingLayer;

/* Optimizer prototype */
class Optimizer;

/* An interface for different type of layer in network */
class Layer {
	public:
	Layer();
	virtual ~Layer();
	
	// share methoid between layer classes
	virtual void propagateForward(std::vector<Matrix*> * input) = 0;
	virtual void propagateBackward(std::vector<Matrix*> * errors) = 0;
	virtual void updateWeightsAndBiases(int batch_size, Optimizer * optimizer);
	
	// element access in derive classes
	virtual std::vector<Matrix*> &outputRef() = 0;  // access to layer output vector
	virtual std::vector<Matrix*> &inputRef() = 0;	// access to layer input vector
};

/* Base class prototype declaration */
class Optimizer 
{
    public:
	Optimizer();
    virtual ~Optimizer();
    virtual void DenseOptimizer(DenseLayer * layer, int batch_size) = 0;
    virtual void ConvOptimizer(ConvolutionalLayer * layer, int batch_size) = 0;
	virtual void ScheduleLearningRate() = 0;
	virtual Scalar getLearningRate() = 0;
	virtual void setLearningRate(Scalar _new_lr) = 0;
};

class LearningRateScheduler {
	public:
    Scalar lr;
    LearningRateScheduler();
	LearningRateScheduler(Scalar _lr);
    virtual ~LearningRateScheduler();

	Scalar getLearningRate();
	virtual void updateLearningRate();
	virtual void resetStep();
	virtual void setLearningRate(Scalar _new_lr);
};

/* An interface for differnt type of layer configuration in network */
class LayerConfig {
	public:
	std::string layerType;
	LayerConfig();
	virtual ~LayerConfig();
	
	// share method between layer classes
	virtual uint32_t &inputHeightRef();
	virtual uint32_t &inputWidthRef();
	virtual uint32_t &inputDepthRef();
	
	virtual uint32_t &kernelHeightRef();
	virtual uint32_t &kernelWidthRef();
	virtual uint32_t &kernelDepthRef();
	
	virtual uint32_t &outputHeightRef();
	virtual uint32_t &outputWidthRef();
	virtual uint32_t &outputDepthRef();

	virtual ScalarFunPtr &activationFunctionRef();
	virtual ScalarFunPtr &activationFunctionDerivativeRef();
	// element access in derive class
};

class DisableMethod : public std::exception {
	std::string msg;
	public:
	DisableMethod(std::string method) {
		msg = "Called disable method: "  + method + '\n';
	}
	const char* what() const throw() {
		return msg.c_str();
	}
};

class UnknownLayerType : public std::exception {
	std::string msg;
	public:
	UnknownLayerType(std::string layerType) {
		msg = "Unknown Layer Type: "  + layerType + '\n';
	}
	const char* what() const throw() {
		return msg.c_str();
	}
};

#endif