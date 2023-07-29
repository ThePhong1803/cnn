#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#include <common.h>
#pragma once

// use typedefs for future ease for changing data types like : float to double
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
typedef uint32_t uint;

// activation fucntion prototype

Scalar Sigmoid(Scalar x);
Scalar dSigmoid(Scalar x);
Scalar ReLU(Scalar x);
Scalar dReLU(Scalar x);
Scalar LeakyReLU(Scalar x);
Scalar dLeakyReLU(Scalar x);
Scalar tanhAct(Scalar x);
Scalar dtanhAct(Scalar x);

// neural network implementation class!
class NeuralNetwork {
public:
	// constructor
	NeuralNetwork(std::vector<uint> topology, std::vector<std::pair<Scalar (*)(Scalar), Scalar (*)(Scalar)>> actFunPtr, Scalar learningRate = Scalar(0.005));
	~NeuralNetwork();

	// function for forward propagation of data
	void propagateForward(RowVector& input);

	// function for backward propagation of errors made by neurons
	void propagateBackward(RowVector& output);

	// function for update weights and bias matrices
	void updateWeightsAndBias(int batchSize);

	// function to train the neural network give an array of data points and return MSE (RMSE acttualy)
	std::pair<float, float> train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data, int (*outputToLabelIdx)(RowVector*), int batchSize);

	// function to validate after each training batch
	std::pair<float, float> validateTrain(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data, int batchSize, int num, int (*outputToLabelIdx)(RowVector*));

	// function to validate the neural network, return the accuracy of the model
	std::pair<float, float> validate(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data, int batchSize, int num, int (*outputToLabelIdx)(RowVector*));
	// storage objects for working of neural network
	/*
		use pointers when using std::vector<Class> as std::vector<Class> calls destructor of
		Class as soon as it is pushed back! when we use pointers it can't do that, besides
		it also makes our neural network class less heavy!! It would be nice if you can use
		smart pointers instead of usual ones like this
		*/
	std::vector<RowVector*> 		neuronLayers; // stores the different layers of out network
	std::vector<RowVector*> 		cachesLayers; // stores the unactivated (activation fn not yet applied) values of layers
	std::vector<std::pair<Scalar (*)(Scalar), Scalar (*)(Scalar)>> 	actFunPtr; //  // stores function pointer for activation of each layer except input layer
	std::vector<Matrix*> 			weights; 	  // the connection weights itself
	std::vector<Matrix*> 			dweights;	  // store the average change of weights matrix value, and will be apply to the weights matrix after a batch training
	std::vector<RowVector*> 		bias;		  // store the bias value for all layers except input layers
	std::vector<RowVector*> 		dbias;		  // store the average change of bias matrix value for all layers except input layers
	std::vector<uint> 				topology;	  // network topology configuraton vector
	Scalar learningRate;
};

#endif