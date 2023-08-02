#include <neuralnetwork.h>
// constructor of neural network class
/*
	@param topology: stl vector input for network topology
	@param learningRate: define how fast the network learn
	*/
NeuralNetwork::NeuralNetwork(std::vector<uint> topology, std::vector<std::pair<Scalar (*)(Scalar), Scalar (*)(Scalar)>> actFunPtr, Scalar learningRate)
{
	this -> topology 		= topology;
	this -> learningRate 	= learningRate;
	// init network
	for(size_t i = 0; i < this -> topology.size(); i++){
		
		/* create neuron layers and caches layer*/
		this -> neuronLayers.push_back(new RowVector(this -> topology[i]));
		this -> neuronLayers.back() -> setZero();
		/* create and init weights matrix */
		if (i > 0) {
			// add new vector to the container
			this -> weights.push_back		(new Matrix		(topology[i - 1], topology[i]	));
			this -> dweights.push_back		(new Matrix		(topology[i - 1], topology[i]	));
			this -> cachesLayers.push_back	(new RowVector	(this -> topology[i]			));	// i.e no caches layers for input	
			this -> bias.push_back			(new RowVector	(this -> topology[i]			));	// i.e no bias for input layers
			this -> dbias.push_back			(new RowVector	(this -> topology[i]			));	// i.e no bias for input layers
			this -> actFunPtr.push_back		(actFunPtr[i - 1]);
			
			// initialization
			this -> cachesLayers.back() -> setZero();
			this -> bias.back()			-> setZero();
			this -> weights.back() 		-> setRandom();
			this -> dbias.back()		-> setZero();
			this -> dweights.back()		-> setZero();
		}
	}
};

// deconstructor of neural network class
NeuralNetwork::~NeuralNetwork()
{
	// iterate each layer, deallocate used memory
	try{
		for(size_t i = 0; i < this -> topology.size(); i++){
			delete this -> neuronLayers.back();
			this -> neuronLayers.pop_back();
			
			if(i > 0){
				delete this -> cachesLayers.back();
				delete this -> weights.back();
				delete this -> dweights.back();
				delete this -> bias.back();
				delete this -> dbias.back();

				this -> cachesLayers.pop_back();
				this -> weights.pop_back();
				this -> dweights.pop_back();
				this -> bias.pop_back();
				this -> dbias.pop_back();
			}
		}
	} catch(std::exception &e){
		std::cout << e.what() << std::endl;
		exit(-1);
	}
};
void NeuralNetwork::propagateForward(RowVector& input)
{
	// set the input to input layer
    // block returns a part of the given vector or matrix
    // block takes 4 arguments : startRow, startCol, blockRows, blockCols
	(*neuronLayers.front()) = input;
	for (uint i = 0; i < topology.size() - 1; i++) {
		// calculate the next layer caches valued
		(*cachesLayers[i]) = (*neuronLayers[i]) * (*weights[i]) + (*bias[i]);
		// propagate the data forward and then
		// apply the activation function to your network
		// unaryExpr applies the given function to all elements of CURRENT_LAYER
		(*neuronLayers[i + 1]) = cachesLayers[i] -> unaryExpr([this, i](Scalar x) {return this -> actFunPtr[i].first(x);});
    }
}

void NeuralNetwork::propagateBackward(RowVector& output)
{
	// calc output error
	/*
		For output layers error calculation:
		@param: output is the desire output
		@param: (*neuronLayers[neuronLayers.size() - 1) is the activated output of the network or predicted output
		#param: (*cachesLayers(cachesLayers.size() - 1) is the unactivated output of the network or weighted value;
		*/
	RowVector errors = (output - (*neuronLayers[neuronLayers.size() - 1]));
	for (int i = weights.size() - 1; i >= 0; i--) {
		// calculate delta of current layer
		Matrix delta = (errors.array() * (cachesLayers[i] -> unaryExpr([this, i](Scalar x) {return this -> actFunPtr[i].second(x);})).array());
		// we need to error for prev layer before update weight and bias
		errors = errors * weights[i] -> transpose();		// Cross entropy optimization
		(*dbias[i]) 	+= delta;
		(*dweights[i]) 	+= (neuronLayers[i] -> transpose() * delta);
		// we need to pass error value back to prev layer
    }
}

void NeuralNetwork::updateWeightsAndBias(int batchSize) {
	// update weight and bias
	for(uint i = 0; i < weights.size(); i++){
		// apply learning rate
		(*dweights[i]) 	*= this -> learningRate;
		(*dbias[i])	  	*= this -> learningRate;

		// update the network weights and biases
		(*weights[i]) 	+= ((*dweights[i]) / float(batchSize));  // add the average change in weight
		(*bias[i])	  	+= ((*dbias[i])    / float(batchSize));  // add the average change in bias

		// after update weights and bias matrix, reinit dweights and dbias to zero
		this -> dbias[i]		-> setZero();
		this -> dweights[i]		-> setZero();
	}
}

std::pair<float, float> NeuralNetwork::train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data, int (*outputToLabelIdx)(RowVector*), int batchSize)
{
	float MSE = 0;
	float ACC = 0;
	// init dweight and dbias matrices zero every training batch
	
	for (int i = 0; i < batchSize; i++) {
		int index = rand() % batchSize;
		//this -> printNetwork();
		// std::cout << "___________________________________________" << std::endl;
        // std::cout << "Input to neural network is : " << *input_data[index] << std::endl;
        propagateForward(*input_data[index]);
        // std::cout << "Expected output is : " << *output_data[index] << std::endl;
        propagateBackward(*output_data[index]);
		// std::cout << "Actual output is : " << *neuronLayers.back() << std::endl;
        // std::cout << "MSE : " << std::sqrt((*neuronLayers.back() - *output_data[index]).dot((*neuronLayers.back() - *output_data[index])) / neuronLayers.back()->size()) << std::endl;
		// std::cout << "___________________________________________" << std::endl;
		// this -> printNetwork();
		MSE += 0.5f * (*neuronLayers.back() - *output_data[index]).dot((*neuronLayers.back() - *output_data[index]));
		int output_num = outputToLabelIdx(neuronLayers.back());
		int expected_num = outputToLabelIdx(output_data[index]);
		if(output_num == expected_num) ACC++;
    }
	// update weights and biases
	this -> updateWeightsAndBias(batchSize);
	MSE /= batchSize;
	ACC /= batchSize;
	return std::pair<float, float>(MSE, ACC);
}

std::pair<float, float> NeuralNetwork::validate(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data, int batchSize, int num, int (*outputToLabelIdx)(RowVector*))
{
	/*
		This function take the load test dataset, and validate the model performance,
		it return a pair of float number, which is average accurage and average confident level
		*/
	float ACC = 0;
	float CON = 0;
	for (int i = 0; i < num; i++) {
		/*
			Take one random element from test data and desired output data, out they are match, increase accuracy counter;
			*/
		std::cout << "===================================================================" << std::endl;
		int index = rand() % batchSize;
        propagateForward(*input_data[index]);
		int output_num   = outputToLabelIdx(neuronLayers.back());
		int expected_num = outputToLabelIdx(output_data[index]);
		float confidence = neuronLayers.back() -> coeff(output_num);
		std::cout << "Actual output      : "<< output_num << " Confident: " << std::fixed << std::setprecision(2) << confidence << std::endl;
		std::cout << "Expected output is : " << expected_num << std::endl;
		std::cout << "Output vector:    " << *neuronLayers.back() << std::endl;
		std::cout << "Expected vector : " << *output_data[index] << std::endl;
		std::cout << "===================================================================" << std::endl;
		
		// update average accuracy and confident level
		if(output_num == expected_num) {
			ACC++;
			CON += confidence;
		}
    }
	ACC /= num;
	CON /= num;
	return std::pair<float, float>(ACC, CON);
}

std::pair<float, float> NeuralNetwork::validateTrain(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data, int batchSize, int num, int (*outputToLabelIdx)(RowVector*))
{
	/*
		This function take the load test dataset, and validate the model performance,
		it return a pair of float number, which is average accurage and average confident level
		*/
	float ACC = 0;
	float MSE = 0;
	// float CON = 0;
	for (int i = 0; i < num; i++) {
		/*
			Take one random element from test data and desired output data, out they are match, increase accuracy counter;
			*/
		int index = rand() % batchSize;
        propagateForward(*input_data[index]);
		int output_num = outputToLabelIdx(neuronLayers.back());
		int expected_num = outputToLabelIdx(output_data[index]);
		//float confidence = neuronLayers.back() -> coeff(output_num);
		MSE += 0.5f * (*neuronLayers.back() - *output_data[index]).dot((*neuronLayers.back() - *output_data[index]));
		// update accuracy and confident level
		if(output_num == expected_num) {
			ACC++;
		}
		
    }
	ACC /= num;
	MSE /= num;
	return std::pair<float, float>(ACC, MSE);
}