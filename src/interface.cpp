#include <interface.h>

/* Begin Layer interface class section */

Layer::Layer()
{
	// Do nothing
}

Layer::~Layer()
{
	// Do nothing, this is virtual destructor so let derive class destructor do the work
}

/*	End Layer interface class section */

/* Begin LayerConfig interface class section */
LayerConfig::LayerConfig()
{
	// Do nothing
}

LayerConfig::~LayerConfig()
{
	// Do nothing, this is virtual destructor so let derive class destructor do the work
}

void Layer::updateWeightsAndBiases(int batch_size, Optimizer * optimizer)
{
	// Do nothing
}

/* Optimizer interface class */
/* Base class */
Optimizer::Optimizer()
{
    // Do nothing
}

Optimizer::~Optimizer()
{
    // Base class virutal destructor
}

LearningRateScheduler::LearningRateScheduler()
{
	this -> lr = 0.01; // default value of learning rate base class
}

LearningRateScheduler::LearningRateScheduler(Scalar _lr)
{
	this -> lr = _lr; // default value of learning rate base class
}

LearningRateScheduler::~LearningRateScheduler()
{
	// Do nothing
}

Scalar LearningRateScheduler::getLearningRate()
{
	// return model current learning rate
	return this -> lr;
}

void LearningRateScheduler::setLearningRate(Scalar _new_lr)
{
	// return model current learning rate
	this -> lr = _new_lr;
}

void LearningRateScheduler::updateLearningRate()
{
	// Do nothing - overrided method
	throw DisableMethod("updateLearningRate");
}

void LearningRateScheduler::resetStep()
{
	// Do nothing - overrided method
	throw DisableMethod("resetStep");
}
/* Base class method is not callable */
void Optimizer::DenseOptimizer(DenseLayer * layer, int batch_size)
{
    throw DisableMethod("DenseOptimizer");
}
void Optimizer::ConvOptimizer(ConvolutionalLayer * layer, int batch_size)
{
    throw DisableMethod("ConvOptimizer");
}

uint32_t &LayerConfig::inputHeightRef()
{
	throw DisableMethod("inputHeightRef");
}

uint32_t &LayerConfig::inputWidthRef()
{
	throw DisableMethod("inputWidthRef");
}

uint32_t &LayerConfig::inputDepthRef()
{
	throw DisableMethod("inputDepthRef");
}

uint32_t &LayerConfig::outputHeightRef()
{
	throw DisableMethod("inputHeightRef");
}

uint32_t &LayerConfig::outputWidthRef()
{
	throw DisableMethod("inputWidthRef");
}

uint32_t &LayerConfig::outputDepthRef()
{
	throw DisableMethod("inputDepthRef");
}

uint32_t &LayerConfig::kernelHeightRef()
{
	throw DisableMethod("kernelHeightRef");
}

uint32_t &LayerConfig::kernelWidthRef()
{
	throw DisableMethod("kernelWidthRef");
}

uint32_t &LayerConfig::kernelDepthRef()
{
	throw DisableMethod("kernelDepthRef");
}

ScalarFunPtr &LayerConfig::activationFunctionRef()
{
	throw DisableMethod("activationFunctionRef");
}
ScalarFunPtr &LayerConfig::activationFunctionDerivativeRef()
{
	throw DisableMethod("activationFunctionDerivativeRef");
}
/* End LayerConfig interface class section */