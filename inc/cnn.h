#ifndef __CNN_H__
#define __CNN_H__

#include <common.h>
#include <convolutionallayer.h>
#include <poolinglayer.h>
#include <flattenlayer.h>

// create class prototype
class ConvolutionalNeuralNetwork
{
	public:
	std::vector<Layer *> 		layer;	// a vector container for all layer in this neural network
	std::vector<LayerConfig *> 	config;
	
	public:
	std::vector<Matrix *> 		inputRef();
	std::vector<Matrix *> 		outputRef();
	
	ConvolutionalNeuralNetwork(std::vector<LayerConfig *> _config);
	~ConvolutionalNeuralNetwork();
	
	void propagateForward(std::vector<Matrix*> input);
	void propagateBackward(std::vector<Matrix*> errors);
	
	private:
	void deleteNetwork();
};

#endif