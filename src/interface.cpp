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

void Layer::updateWeightsAndBiases(int batch_size)
{
	// Do nothing
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