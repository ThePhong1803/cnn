#include <common.h>
#include <neuralnetwork.h>
#include <cnn.h>

#define MATLEN 10
#define KERNEL 3
#define POOL   3

// #define TESTING

#ifndef TESTING
int main(int argc, char **argv) {
	assert(argc == 2);
	uint32_t epoch = atoi(argv[1]);
	srand(time(NULL));
	// prepare image data matrix
	// ImageData img("test-img/test.bmp");
	// Matrix mat(img.height, img.width);
	// img.getPixelMatrix(&mat);
	// std::vector<Matrix*> vec;
	// vec.push_back(&mat);

	std::vector<Matrix*> vec;
	vec.push_back(new Matrix(MATLEN, MATLEN));
	for(int i = 0; i < MATLEN; i++){
		for(int j = 0; j < MATLEN; j++){
			vec.back() -> coeffRef(i, j) = ((double) rand() / (RAND_MAX));
		}
	}
	vec.push_back(new Matrix(MATLEN, MATLEN));
	for(int i = 0; i < MATLEN; i++){
		for(int j = 0; j < MATLEN; j++){
			vec.back() -> coeffRef(i, j) = ((double) rand() / (RAND_MAX));
		}
	}
	
	// config vector
	std::vector<LayerConfig *> config;

	// config convolutional layer
	ConvConfig config_0;
	config_0.layerType	  = "conv";
	config_0.inputHeight  = MATLEN;
	config_0.inputWidth   = MATLEN;
	config_0.inputDepth   = 2;
	config_0.kernelHeight = KERNEL;	// hyperparameter
	config_0.kernelWidth  = KERNEL;	// hyperparameter
	config_0.numKernel    = 4;		// hyperparameter
	config_0.padding      = 0;		// hyperparameter, usually zero
	config_0.striding     = 1;		// hyperparameter ?? (unsure, use default)
	config_0.actFun		  = ReLU;
	config_0.dactFun	  = dReLU;
	config_0.learningRate = 0.01;

	ConvConfig config_1;
	config_1.layerType		= "conv";
	config_1.inputHeight  	= MATLEN - KERNEL + 1;	
	config_1.inputWidth   	= MATLEN - KERNEL + 1;	
	config_1.inputDepth   	= 4;		// depend on previous hyperparameter
	config_1.kernelHeight 	= KERNEL;	// hyperparameter
	config_1.kernelWidth  	= KERNEL;	// hyperparameter
	config_1.numKernel    	= 2;		// hyperparameter
	config_1.padding      	= 0;		// hyperparameter, usually zero
	config_1.striding     	= 1;		// hyperparameter ??? (unsure, use default)
	config_1.actFun			= Sigmoid;
	config_1.dactFun		= dSigmoid;
	config_1.learningRate 	= 0.01;

	PoolingConfig config_2;
	config_2.layerType		= "maxpool";
	config_2.inputHeight  	= 6;	
	config_2.inputWidth   	= 6;	
	config_2.inputDepth   	= 2;	// depend on previous hyperparameter
	config_2.kernelHeight 	= 2;	// hyperparameter, must be divisible by input widen
	config_2.kernelWidth  	= 2;	// hyperparameter, must be divisible by input width

	FlattenConfig config_3;
	config_3.layerType	= "flatten";
	config_3.inputHeight = 3;
	config_3.inputWidth = 3;
	config_3.inputDepth = 2;
	
	config.push_back(&config_0);
	config.push_back(&config_1);
	config.push_back(&config_2);
	config.push_back(&config_3);
	
	std::vector<Matrix *> expected;
	expected.push_back(new Matrix(1, 18));
	*expected[0] << 0, 1, 0,
					0, 1, 0,
					0, 1, 0,
					0, 0, 0,
					1, 1, 1,
					0, 0, 0;
	
	ConvolutionalNeuralNetwork cnn(config);
	
	std::cout << "Input: " << std::endl;
	for(size_t i = 0; i < vec.size(); i++){
		std::cout << "Input Layer: " << i << std::endl;
		std::cout << *vec[i] << std::endl;
	}
	for(uint32_t i = 0; i < epoch; i++){
		cnn.propagateForward(vec);
		cnn.propagateBackward(expected);
		std::cout <<"\rEpoch: " << i + 1; 
	}

	std::cout << "\n===================================" << std::endl;
	std::cout << "Prediction: " << std::endl;
	cnn.propagateForward(vec);
	for(size_t i = 0; i < cnn.outputRef().size(); i++){
		std::cout << "Output Layer: " << i << std::endl;
		std::cout << std::fixed << std::setprecision(2);
		for(size_t r = 0; r < config_3.inputHeight * config_3.inputDepth; r++)
		{
			for(size_t c = 0; c < config_3.inputWidth; c++)
			{
				std::cout << cnn.outputRef()[i] -> coeff(r * config_3.inputWidth + c) << " ";
			}
			std::cout << std::endl;
		}
	}
	std::cout << "=====================================" << std::endl;
	for(size_t i = 0; i < cnn.layer.size(); i++){
		std::cout << "Output Layer: " << config[i] -> layerType << std::endl;
		std::cout << std::fixed << std::setprecision(2);
		for(size_t j = 0; j < cnn.layer[i] -> outputRef().size(); j++)
		{
		std::cout << *cnn.layer[i] -> outputRef()[j] << std::endl;
		}
	}

	while(expected.size() != 0){
		delete expected.back();
		expected.pop_back();
	}
	while(vec.size() != 0){
		delete vec.back();
		vec.pop_back();
	}
	return 0;
}

#else 

int main()
{
	Matrix mat0(3, 3), mat1(3,3);
	mat0 << 1,2,3,4,5,6,7,8,9;
	mat1 << 10,11,12,13,14,15,16,17,18;
	Matrix * error = new Matrix(1, 18);
	*error << 1,  4,  7,  2,  5,  8,  3,  6,  9, 10, 13, 16 ,11, 14 ,17 ,12, 15, 18;
	std::cout << mat0 << std::endl;
	std::cout << mat1 << std::endl;
	std::vector<Matrix *> input;
	std::vector<Matrix *> errors;
	errors.push_back(error);
	input.push_back(&mat0);
	input.push_back(&mat1);
	FlattenConfig config;
	config.inputHeight = 3;
	config.inputWidth = 3;
	config.inputDepth = 2;

	FlattenLayer layer(&config);
	layer.propagateForward(&input);
	layer.propagateBackward(&errors);
	std::cout << *layer.outputRef().back() << std::endl;
	for(size_t i = 0; i < errors.size(); i++)
	{
		std::cout << *errors[i] << std::endl;
	}
	return 0;
}

#endif