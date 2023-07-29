#include <cnn.h>

/* convolutional neuralnetwork implementation */

// convolutional neuralnetwork consturctor
ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(std::vector<LayerConfig *> _config) : config(_config) {
	for(size_t i = 0; i < config.size(); i++)
	{
		if		(config[i] -> layerType == "conv") {
			this -> layer.push_back(new ConvolutionalLayer((ConvConfig*)config[i]));
		}
		else if (config[i] -> layerType == "maxpool") {
			this -> layer.push_back(new MaxPoolingLayer((PoolingConfig*)config[i]));
		}
		else if (config[i] -> layerType == "flatten") {
			this -> layer.push_back(new FlattenLayer((FlattenConfig*)config[i]));
		}
		else {
			// clean object instance before exit
			this -> deleteNetwork();
			throw UnknownLayerType(config[i] -> layerType);
		}
	}
	
	// link a layer output to next layer output
	for(size_t i = 1; i < layer.size(); i++)
	{
		layer[i] -> inputRef() = layer[i - 1] -> outputRef();
	}
}

// convolutional neuralnetwork destructor
ConvolutionalNeuralNetwork::~ConvolutionalNeuralNetwork() {
	this -> deleteNetwork();
}

// method to delete this class instance
void ConvolutionalNeuralNetwork::deleteNetwork()
{
	while(this -> layer.size() != 0)
	{
		delete this -> layer.back();
		this -> layer.pop_back();
	}
}

std::vector<Matrix *> ConvolutionalNeuralNetwork::inputRef()
{
	return this -> layer.front() -> inputRef();
}

std::vector<Matrix *> ConvolutionalNeuralNetwork::outputRef()
{
	return this -> layer.back() -> outputRef();
}

// Implement convolutional neural network propagate forward method
void ConvolutionalNeuralNetwork::propagateForward(std::vector<Matrix *> input)
{
	layer.front() -> inputRef() = input;
	for(size_t i = 0; i < layer.size(); i++)
	{
		layer[i] -> propagateForward(&layer[i] -> inputRef());
	}
}

// Implement convolutional nerual network propagate backward method
void ConvolutionalNeuralNetwork::propagateBackward(std::vector<Matrix*> expected)
{
	// calculate error matrix
	std::vector<Matrix *> errors;
	for(size_t i = 0; i < layer.back() -> outputRef().size(); i++)
	{
		errors.push_back(new Matrix(expected[i] -> rows(), expected[i] -> cols()));
		*errors[i] = *expected[i] - *layer.back() -> outputRef()[i];
	}

	// propagate the errors back to all hidden layer
	for(int i = (int)layer.size() - 1; i >= 0; i--)
	{
		layer[i] -> propagateBackward(&errors);
	}
	// delete error matrix vector
	while(errors.size() != 0)
	{
		delete errors.back();
		errors.pop_back();
	}
}