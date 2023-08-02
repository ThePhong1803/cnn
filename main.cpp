#include <common.h>
#include <imagedata.h>
#include <neuralnetwork.h>
#include <cnn.h>
#include <filesystem>

#define DATASIZE 60000
#define TESTSIZE 10000
#define TEST     100
#define TESTING

void loadDataset(   std::vector<std::vector<Matrix *>> &input_data,             // input data
                    std::vector<std::vector<Matrix *>> &output_data,            // output data
                    std::vector<std::vector<Matrix>> &targetOutputs,          // target output for mapping
                    std::vector<int*>       &labelVec,               // label vector for mapping
                    int datasize                                        // batch size of the training
                )
{
    /* - Loading image data into data container */
	int stIdx = rand() % (DATASIZE - datasize);
	char buff[128];
	for(int n = 0; n < datasize; n++){
		try{
			sprintf(buff, "dataset/training_dataset/%05d.bmp", stIdx + n + 1);
			ImageData img(buff);
			Matrix * tmp = new Matrix(img.height, img.width);
			img.getPixelMatrix(tmp);
			input_data.push_back(std::vector<Matrix *>());
			output_data.push_back(std::vector<Matrix *>());
			input_data.back().push_back(tmp);
			output_data.back().push_back(&targetOutputs[*labelVec[stIdx + n]].back());
		} catch (std::exception &e) {
			std::cout << e.what() << std::endl;
		}
	}
}

void loadTestData(  std::vector<std::vector<Matrix *>> &input_data,             // input data
                    std::vector<std::vector<Matrix *>> &output_data,            // output data
                    std::vector<std::vector<Matrix>> &targetOutputs,          // target output for mapping
                    std::vector<int*>       &labelVec,               // label vector for mapping
                    int datasize                                        // batch size of the training
                )
{
    /* - Loading image data into data container */
	int stIdx = rand() % (TESTSIZE - datasize);
	char buff[128];
	for(int n = 0; n < datasize; n++){
		try{
			sprintf(buff, "dataset/test_dataset/%05d.bmp", stIdx + n + 1);
			ImageData img(buff);
			Matrix * tmp = new Matrix(img.height, img.width);
			img.getPixelMatrix(tmp);
			input_data.push_back(std::vector<Matrix *>());
			output_data.push_back(std::vector<Matrix *>());
			input_data.back().push_back(tmp);
			output_data.back().push_back(&targetOutputs[*labelVec[stIdx + n]].back());
		} catch (std::exception &e) {
			std::cout << e.what() << std::endl;
		}
	}
}

void cleanDataBuffer(std::vector<std::vector<Matrix *>> &input_data) {
	// deallocate memory used for input and output
	while(input_data.size() != 0){
		while(input_data.back().size() != 0){
			delete input_data.back().back();
			input_data.back().pop_back();
		}
		input_data.pop_back();
	}
}
void cleanLabelBuffer(std::vector<std::vector<Matrix *>> &input_data) {
	// deallocate memory used for input and output
	while(input_data.size() != 0){
		while(input_data.back().size() != 0){
			input_data.back().pop_back();
		}
		input_data.pop_back();
	}
}

int outputToLabelIdx(Matrix * out){
	// Sigmoid or ReLU always produce output greater than 0
    float max = 0;
	int idx = 0;
	for(int i = 0; i < out -> cols(); i++){
		if(out -> coeff(0 ,i) > max){
			max = out -> coeff(i);
			idx = i;
		}
	}
	return idx;
}
// #define TESTING

#ifndef TESTING
int main(int argc, char ** argv)
{
	// Reading input argument
	// std::filesystem::current_path("C:/Users/PC/Desktop/NNProject");
	srand(time(NULL));
	if(argc < 4) {
		std::cout << "Using: out --learning-rate --epoch --batch" << std::endl;
		exit(-1);
	}
	// float learnRate = atof(argv[1]);
	float learningRate	= atof(argv[1]);
	int epoch		    = atoi(argv[2]);
	int batchSize	    = atoi(argv[3]);
	std::cout << "Epoch: " << epoch << std::endl;
	std::cout << "Batch size: " << batchSize << std::endl;

	// loading dataset
	/* - Fist is setup input data containter, then the expected output data container */
	std::vector<std::vector<Matrix *>> input_data;
	std::vector<std::vector<Matrix *>> output_data;
	std::vector<std::vector<Matrix>>   targetOutputs(10, std::vector<Matrix>(1, Matrix(1, 10))); 
	
    /* - Create labeled array for mapping */
    std::vector<int*> labelVec;
    std::ifstream labelFile;
    labelFile.open("dataset/training_dataset/label.txt", std::ios::in | std::ios::binary);
    if(!labelFile.is_open()) {
        std::cout << "File not found" << std::endl;
        exit(-1);
    }

    int label;
    while(labelFile >> label){
        labelVec.push_back(new int(label));
    }
	labelFile.close();   

	std::vector<int*> validateLabels;
    std::ifstream validateLabelsFile;
    validateLabelsFile.open("dataset/test_dataset/00000-labels.txt");
    if(!validateLabelsFile.is_open()) {
        std::cout << "File not found" << std::endl;
        exit(-1);
    }

    int testLabel;
    while(validateLabelsFile >> testLabel){
        validateLabels.push_back(new int(testLabel));
    }
	validateLabelsFile.close(); 

	/* - Mapping the expected output data to the output data container */
	for(int i = 0; i < 10; i++){
		for(int j = 0; j < 10; j++){
			if(i == j){
				targetOutputs[i][0].coeffRef(0, j) = 1.0f;
			}
			else {
				targetOutputs[i][0].coeffRef(0, j) = 0.0f;
			}
		}
	}
	/* Configuration Convolutional neuralnetwork */

	// config vector
	std::vector<LayerConfig *> config;

	// config convolutional layer
	ConvConfig config_0;
	config_0.layerType	  = "conv";
	config_0.inputHeight  = 28;
	config_0.inputWidth   = 28;
	config_0.inputDepth   = 1;
	config_0.kernelHeight = 3;	// hyperparameter
	config_0.kernelWidth  = 3;	// hyperparameter
	config_0.numKernel    = 32;		// hyperparameter
	config_0.padding      = 1;		// hyperparameter, usually zero
	config_0.striding     = 1;		// hyperparameter ?? (unsure, use default)
	config_0.actFun		  = tanhAct;
	config_0.dactFun	  = dtanhAct;
	config_0.learningRate = learningRate;

	PoolingConfig config_1;
	config_1.layerType		= "maxpool";
	config_1.inputHeight  	= 28;	
	config_1.inputWidth   	= 28;	
	config_1.inputDepth   	= 32;	// depend on previous hyperparameter
	config_1.kernelHeight 	= 2;	// hyperparameter, must be divisible by input widen
	config_1.kernelWidth  	= 2;	// hyperparameter, must be divisible by input width

	ConvConfig config_2;
	config_2.layerType		= "conv";
	config_2.inputHeight  	= 14;	 
	config_2.inputWidth   	= 14;	
	config_2.inputDepth   	= 32;		// depend on previous hyperparameter
	config_2.kernelHeight 	= 3;	// hyperparameter
	config_2.kernelWidth  	= 3;	// hyperparameter
	config_2.numKernel    	= 64;		// hyperparameter
	config_2.padding      	= 1;		// hyperparameter, usually zero
	config_2.striding     	= 1;		// hyperparameter ??? (unsure, use default)
	config_2.actFun			= tanhAct;
	config_2.dactFun		= dtanhAct;
	config_2.learningRate 	= learningRate;

	PoolingConfig config_3;
	config_3.layerType		= "maxpool";
	config_3.inputHeight  	= 14;	
	config_3.inputWidth   	= 14;	
	config_3.inputDepth   	= 64;	// depend on previous hyperparameter
	config_3.kernelHeight 	= 2;	// hyperparameter, must be divisible by input widen
	config_3.kernelWidth  	= 2;	// hyperparameter, must be divisible by input width

	FlattenConfig config_4;
	config_4.layerType	= "flatten";
	config_4.inputHeight = 7;
	config_4.inputWidth = 7;
	config_4.inputDepth = 64;

	DenseConfig config_5;
	config_5.layerType = "dense";
	config_5.inputWidth = 3136;
	config_5.outputWidth = 512;
	config_5.actFun = Sigmoid;
	config_5.dactFun = dSigmoid;
	config_5.learningRate = learningRate;

	DenseConfig config_6;
	config_6.layerType = "dense";
	config_6.inputWidth = 512;
	config_6.outputWidth = 256;
	config_6.actFun = Sigmoid;
	config_6.dactFun = dSigmoid;
	config_6.learningRate = learningRate;

	SoftmaxConfig config_7;
	config_7.layerType = "softmax";
	config_7.inputWidth = 256;
	config_7.outputWidth = 10;
	config_7.learningRate = learningRate;
	
	config.push_back(&config_0);
	config.push_back(&config_1);
	config.push_back(&config_2);
	config.push_back(&config_3);
	config.push_back(&config_4);
	config.push_back(&config_5);
	config.push_back(&config_6);
	config.push_back(&config_7);

	ConvolutionalNeuralNetwork cnn(config);

	/* - Training with loaded data */
	std::ofstream log("./log/RMSE.txt");
	log << "RMSE" << " " << "ACC" << '\n';
	std::cout << std::fixed << std::setprecision(4);
	for(int i = 0; i < epoch; i++){
		srand(time(NULL));
		/* Train and validate after each batch*/
        loadDataset(input_data, output_data, targetOutputs, labelVec, batchSize);
		Scalar MSE = cnn.train(input_data, output_data, batchSize);
		cleanDataBuffer(input_data);
		cleanLabelBuffer(output_data);
		loadTestData(input_data, output_data, targetOutputs, validateLabels, TEST);
		Scalar ACC = cnn.validate(input_data, output_data, outputToLabelIdx, TEST);
		cleanDataBuffer(input_data);
		cleanLabelBuffer(output_data);

		/* Save to log file after each epoch */
		log << MSE << " " << ACC << '\n';

	 	std::cout << "\rEpoch : " << i + 1 << " ACC: " << ACC << " MSE: " << MSE << std::endl;
	}
	std::cout << std::endl;
	log.close();

	/* - Calculate model accuracy */
	loadTestData(input_data, output_data, targetOutputs, validateLabels, 100);
	Scalar accuracy = cnn.validate(input_data, output_data, outputToLabelIdx, 100);
	cleanDataBuffer(input_data);
	cleanLabelBuffer(output_data);
	std::cout << "\rModel average accuracy  : " << accuracy << std::endl;
	std::cout << "===========================================================" << std::endl;

	std::string cmd = "";
	while(cmd != "exit"){
		// get test sample
		Image img("test-img/test.bmp", -1);		// open image for diesplay
		ImageData obj("test-img/test.bmp");		// open image for extract data
        img.setInvert(false);

		Matrix testData(img.getHeight(), img.getWidth()); 	// input data storage
		obj.getPixelMatrix(&testData);						// load pixel data into testing input matrix
		std::vector<Matrix *> test;							// create vector containter for input matrix
		test.push_back(&testData);							
		cnn.propagateForward(test);							// propagate forward testing input matrix vector
		img.testing();										// display test sample
		
		int num = outputToLabelIdx(cnn.layer.back() -> outputRef().back());
		std::cout << "Predicted output: " << num << " Confident: " << std::fixed << std::setprecision(2) << cnn.layer.back() -> outputRef().back() -> coeff(num) << std::endl;
		std::cout << "Output vector: " << *cnn.layer.back() -> outputRef().back() << std::endl;
		getline(std::cin, cmd);
	}
	return 0;
}
#else
int main(int argc, char ** argv){
	
	// srand(time(NULL));
	// // float learnRate = atof(argv[1]);
	// float learningRate = atof(argv[1]);
	// int epoch		   = atoi(argv[2]);
	// int batchSize	   = atoi(argv[3]);
	// std::cout << "Epoch: " << epoch << std::endl;
	// std::cout << "Batch size: " << batchSize << std::endl;

	// // loading dataset
	// /* - Fist is setup input data containter, then the expected output data container */
	// std::vector<std::vector<Matrix *>> input_data;
	// std::vector<std::vector<Matrix *>> output_data;
	// std::vector<std::vector<Matrix>>   targetOutputs(10, std::vector<Matrix>(1, Matrix(1, 10))); 
	
    // /* - Create labeled array for mapping */
    // std::vector<int*> labelVec;
    // std::ifstream labelFile;
    // labelFile.open("dataset/training_dataset/label.txt");
    // if(!labelFile.is_open()) {
    //     std::cout << "File not found" << std::endl;
    //     exit(-1);
    // }

    // int label;
    // while(labelFile >> label){
    //     labelVec.push_back(new int(label));
    // }
	// labelFile.close();   

	// std::vector<int*> validateLabels;
    // std::ifstream validateLabelsFile;
    // validateLabelsFile.open("dataset/test_dataset/00000-labels.txt");
    // if(!validateLabelsFile.is_open()) {
    //     std::cout << "File not found" << std::endl;
    //     exit(-1);
    // }

    // int testLabel;
    // while(validateLabelsFile >> testLabel){
    //     validateLabels.push_back(new int(testLabel));
    // }
	// validateLabelsFile.close(); 

	// /* - Mapping the expected output data to the output data container */
	// for(int i = 0; i < 10; i++){
	// 	for(int j = 0; j < 10; j++){
	// 		if(i == j){
	// 			targetOutputs[i][0].coeffRef(0, j) = 1.0f;
	// 		}
	// 		else {
	// 			targetOutputs[i][0].coeffRef(0, j) = 0.0f;
	// 		}
	// 	}
	// }

	// std::vector<LayerConfig *> config;
	// FlattenConfig config_4;
	// config_4.layerType	= "flatten";
	// config_4.inputHeight = 28;
	// config_4.inputWidth = 28;
	// config_4.inputDepth = 1;

	// DenseConfig config_5;
	// config_5.layerType = "dense";
	// config_5.inputWidth = 784;
	// config_5.outputWidth = 32;
	// config_5.actFun = Sigmoid;
	// config_5.dactFun = dSigmoid;
	// config_5.learningRate = learningRate;

	// DenseConfig config_6;
	// config_6.layerType = "dense";
	// config_6.inputWidth = 32;
	// config_6.outputWidth = 32;
	// config_6.actFun = Sigmoid;
	// config_6.dactFun = dSigmoid;
	// config_6.learningRate = learningRate;

	// DenseConfig config_7;
	// config_7.layerType = "dense";
	// config_7.inputWidth = 32;
	// config_7.outputWidth = 10;
	// config_7.actFun = Sigmoid;
	// config_7.dactFun = dSigmoid;
	// config_7.learningRate = learningRate;
	
	// // config.push_back(&config_0);
	// // config.push_back(&config_1);
	// // config.push_back(&config_2);
	// // config.push_back(&config_3);
	// config.push_back(&config_4);
	// config.push_back(&config_5);
	// config.push_back(&config_6);
	// config.push_back(&config_7);

	// ConvolutionalNeuralNetwork cnn(config);

	// /* - Training with loaded data */
	// std::cout << std::fixed << std::setprecision(2);
	// for(int i = 0; i < epoch; i++){
	// 	/* Train and validate after each batch*/
    //     loadDataset(input_data, output_data, targetOutputs, labelVec, batchSize);
	// 	Scalar MSE = cnn.train(input_data, output_data, batchSize);
	// 	cleanDataBuffer(input_data);
	// 	cleanLabelBuffer(output_data);
	// 	loadTestData(input_data, output_data, targetOutputs, validateLabels, TESTSIZE - 1);
	// 	Scalar ACC = cnn.validate(input_data, output_data, outputToLabelIdx, TESTSIZE - 1);
	// 	cleanDataBuffer(input_data);
	// 	cleanLabelBuffer(output_data);

	//  	std::cout << "\rEpoch : " << i + 1 << " ACC: " << ACC << " MSE: " << MSE << std::endl;
	// 	// std::cout << "\rEpoch : " << i + 1 << " MSE: " << MSE << std::endl;
	// }
	// std::cout << std::endl;

	// Softmax testing proto
	Matrix mat1(4,4);
	mat1.setZero();
	std::cout << "Before: " << &mat1 << std::endl;
	Matrix mat2(1,4);
	mat2 << 1,2,3,4;
	mat1 = mat2.transpose().rowwise().replicate(4);
	std::cout << "After: " << &mat1 << std::endl;
	std::cout << mat1 << std::endl;
}
#endif