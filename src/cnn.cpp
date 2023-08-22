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
		else if (config[i] -> layerType == "dense") {
			this -> layer.push_back(new DenseLayer((DenseConfig*)config[i]));
		}
		else if (config[i] -> layerType == "softmax") {
			this -> layer.push_back(new SoftmaxLayer((SoftmaxConfig*)config[i]));
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

void ConvolutionalNeuralNetwork::summary(){
	for(size_t i = 0; i < config.size(); i++)
	{
		std::cout << "Layer " << config[i] -> layerType << std::endl;
	}
}
// Implement convolutional neural network propagate forward method
void ConvolutionalNeuralNetwork::propagateForward(std::vector<Matrix *> input)
{
	layer.front() -> inputRef() = input;
	for(size_t i = 0; i < layer.size(); i++)
	{
		layer[i] -> propagateForward(&layer[i] -> inputRef());
		// std::cout << "Output Layer " << i << std::endl;
		// for(size_t j = 0; j < layer[i] -> outputRef().size(); j++)
		// {
			// std::cout << "Layer " << j << std::endl;
			// std::cout << *layer[i] -> outputRef()[j] << std::endl;
		// }
	}
}

// Implement convolutional nerual network propagate backward method
void ConvolutionalNeuralNetwork::propagateBackward(std::vector<Matrix*> errors)
{
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

void ConvolutionalNeuralNetwork::updateNetwork(int batchSize) {
	for(size_t i = 0; i < layer.size(); i++){
		layer[i] -> updateWeightsAndBiases(batchSize);
	}
}

// training and validate implementation
Scalar ConvolutionalNeuralNetwork::train(std::vector<std::vector<Matrix *>> input,
										 std::vector<std::vector<Matrix *>> output,
										 int batchSize)
{
	// fist we shuffle train data
	auto rng = std::default_random_engine {};
	std::vector<std::pair<std::vector<Matrix *> *,std::vector<Matrix *> *>> dataset;
	for(size_t i = 0; i < input.size(); i++){
		dataset.push_back(std::pair<std::vector<Matrix *> *,std::vector<Matrix *> *>(&input[i], &output[i]));
	}
	// using random engine to shuffle the dataset
	std::shuffle(std::begin(dataset), std::end(dataset), rng);
	// Divide the input data set to batch with size of (dataset_size / batch)
	// in 1 epoch, we loop throught all batches which assamble the entire dataset
	size_t stIdx = 0;
	Scalar loss = 0;
	for(; stIdx < dataset.size(); stIdx += batchSize)
	{
		/* Loop throught all element in a batch */
		for(int n = 0; n < batchSize && stIdx + n < dataset.size(); n++)
		{
			// propagate forward train data
			this -> propagateForward(*dataset[stIdx + n].first);
			// calculate error matrix
			std::vector<Matrix *> errors = dMeanSquareError(&layer.back() -> outputRef(), dataset[stIdx + n].second);
			// std::cout << "Error: " << *errors.back() << std::endl;
			// calculate loss and propagate back
			loss += CategoricalCrossEntropy(layer.back() -> outputRef().back(), dataset[stIdx + n].second -> back());
			this -> propagateBackward(errors);
		}
		// update start idx
		this -> updateNetwork(batchSize);
		// schedule leanring rate
		std::cout << "\rTrain process: " << float(stIdx) / dataset.size() << " loss: " << loss / (stIdx + batchSize); // clear some char leftover from previous line
	}
	std::cout << "\33[2K\r";
	return loss / dataset.size();
}

Scalar ConvolutionalNeuralNetwork::validate(std::vector<std::vector<Matrix *>> input,
											std::vector<std::vector<Matrix *>> output,
											int (*outputToLabelIdx)(Matrix *), 
											int batchSize)
{
	Scalar ACC = 0;
	for(int n = 0; n < batchSize; n++)
	{
		this -> propagateForward(input[n]);
		// calculate mean square errors
		int output_num   = outputToLabelIdx(layer.back() -> outputRef().back());
		int expected_num = outputToLabelIdx(output[n].back());
		if(output_num == expected_num) ACC++;
		std::cout << "\rValidate process: " << float(n + 1) / batchSize;
	}
	return ACC / batchSize;
}