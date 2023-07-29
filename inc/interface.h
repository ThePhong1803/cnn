#ifndef __LAYER_H__
#define __LAYER_H__
#include <common.h>

/* An interface for different type of layer in network */
class Layer {
	public:
	Layer();
	virtual ~Layer();
	
	// share methoid between layer classes
	virtual void propagateForward(std::vector<Matrix*> * input) = 0;
	virtual void propagateBackward(std::vector<Matrix*> * errors) = 0;
	
	// element access in derive classes
	virtual std::vector<Matrix*> &outputRef() = 0;  // access to layer output vector
	virtual std::vector<Matrix*> &inputRef() = 0;	// access to layer input vector
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