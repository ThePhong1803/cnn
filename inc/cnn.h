#ifndef __CNN_H__
#define __CNN_H__

#include <common.h>
#include <convolutionallayer.h>
#include <poolinglayer.h>
#include <flattenlayer.h>
#include <denselayer.h>
#include <softmaxlayer.h>
#include <losscalc.h>

// create class prototype
class ConvolutionalNeuralNetwork
{
	public:
	std::vector<Layer *> 		layer;	// a vector container for all layer in this neural network
	std::vector<LayerConfig *> 	config;
	
	public:
	std::vector<Matrix *> 		inputRef();
	std::vector<Matrix *> 		outputRef();

	
	// convolutional neural network constructor and destructor
	ConvolutionalNeuralNetwork(std::vector<LayerConfig *> _config);
	~ConvolutionalNeuralNetwork();
	
	// this function perform feed forward for output network
	void propagateForward(std::vector<Matrix*> input);

	// this function perform feed backward for hidden layer in network
	void propagateBackward(std::vector<Matrix*> errors);

	// this function perform update network weights and biases
	void updateNetwork(int batchSize);
	// train and validate neural network
	// train method perform training on network and return the mse value mini batches
	// validate method perform validating on network and return the accuracy of the network
	Scalar train(std::vector<std::vector<Matrix *>> input,std::vector<std::vector<Matrix *>> output, int batchSize);
	Scalar validate(std::vector<std::vector<Matrix *>> input,std::vector<std::vector<Matrix *>> output, int (*outputToLabelIdx)(Matrix *), int batchSize);

	// model summary: print out model structure
	void summary();
	
	private:
	void deleteNetwork();
};

#endif