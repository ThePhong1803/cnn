#include <common.h>
#include <imagedata.h>
#include <cnn.h>

#define DATASIZE 60000
#define TESTSIZE 10000
#define TESTING

Scalar linear(Scalar x) { return x; }
Scalar dlinear(Scalar x) { return 1; }

void loadDataset(   std::vector<std::vector<Matrix *>> &input_data,             // input data
                    std::vector<std::vector<Matrix *>> &output_data,            // output data
                    std::vector<std::vector<Matrix>> &targetOutputs,          // target output for mapping
                    std::vector<int*>       &labelVec,               // label vector for mapping
                    int datasize                                        // batch size of the training
                )
{
    /* - Loading image data into data container */
	char buff[128];
	for(int n = 0; n < datasize; n++){
		try{
			sprintf(buff, "dataset/training_dataset/%05d.bmp", n + 1);
			ImageData img(buff);
			Matrix * tmp = new Matrix(img.height, img.width);
			img.getPixelMatrix(tmp);
			input_data.push_back(std::vector<Matrix *>());
			output_data.push_back(std::vector<Matrix *>());
			input_data.back().push_back(tmp);
			output_data.back().push_back(&targetOutputs[*labelVec[n]].back());
			std::cout << "\rLoading train dataset: " << float(n + 1) / datasize;
		} catch (std::exception &e) {
			std::cout << e.what() << std::endl;
		}
	}
	std::cout << std::endl;
}

void loadTestData(  std::vector<std::vector<Matrix *>> &input_data,             // input data
                    std::vector<std::vector<Matrix *>> &output_data,            // output data
                    std::vector<std::vector<Matrix>> &targetOutputs,          // target output for mapping
                    std::vector<int*>       &labelVec,               // label vector for mapping
                    int datasize                                        // batch size of the training
                )
{
    /* - Loading image data into data container */
	char buff[128];
	for(int n = 0; n < datasize; n++){
		try{
			sprintf(buff, "dataset/test_dataset/%05d.bmp", n + 1);
			ImageData img(buff);
			Matrix * tmp = new Matrix(img.height, img.width);
			img.getPixelMatrix(tmp);
			input_data.push_back(std::vector<Matrix *>());
			output_data.push_back(std::vector<Matrix *>());
			input_data.back().push_back(tmp);
			output_data.back().push_back(&targetOutputs[*labelVec[n]].back());
			std::cout << "\rLoading test dataset: " << float(n + 1) / datasize;
		} catch (std::exception &e) {
			std::cout << e.what() << std::endl;
		}
	}
	std::cout << std::endl;
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

#ifndef TESTING

int main(int argc, char ** argv)
{
	// Reading input argument
	srand(time(NULL));
	if(argc < 3) {
		std::cout << "Using: out --learningRate --epoch --batch" << std::endl;
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
    labelFile.open("dataset/training_dataset/label.txt");
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

	// config convolutional layer, follow lenet configuration, with some variance
	ConvConfig config_0;
	config_0.layerType	  = "conv";
	config_0.inputHeight  = 28;
	config_0.inputWidth   = 28;
	config_0.inputDepth   = 1;
	config_0.kernelHeight = 5;	// hyperparameter
	config_0.kernelWidth  = 5;	// hyperparameter
	config_0.numKernel    = 6;		// hyperparameter
	config_0.padding      = 2;		// hyperparameter, usually zero
	config_0.striding     = 1;		// hyperparameter ?? (unsure, use default)
	config_0.actFun		  = tanhAct;
	config_0.dactFun	  = dtanhAct;
	config_0.learningRate = learningRate;

	PoolingConfig config_1;
	config_1.layerType		= "maxpool";
	config_1.inputHeight  	= 28;	
	config_1.inputWidth   	= 28;	
	config_1.inputDepth   	= 6;	// depend on previous hyperparameter
	config_1.kernelHeight 	= 2;	// hyperparameter, must be divisible by input widen
	config_1.kernelWidth  	= 2;	// hyperparameter, must be divisible by input width

	ConvConfig config_2;
	config_2.layerType		= "conv";
	config_2.inputHeight  	= 14;	 
	config_2.inputWidth   	= 14;	
	config_2.inputDepth   	= 6;		// depend on previous hyperparameter
	config_2.kernelHeight 	= 5;	// hyperparameter
	config_2.kernelWidth  	= 5;	// hyperparameter
	config_2.numKernel    	= 12;		// hyperparameter
	config_2.padding      	= 0;		// hyperparameter, usually zero
	config_2.striding     	= 1;		// hyperparameter ??? (unsure, use default)
	config_2.actFun			= tanhAct;
	config_2.dactFun		= dtanhAct;
	config_2.learningRate 	= learningRate;

	PoolingConfig config_3;
	config_3.layerType		= "maxpool";
	config_3.inputHeight  	= 10;	
	config_3.inputWidth   	= 10;	
	config_3.inputDepth   	= 12;	// depend on previous hyperparameter
	config_3.kernelHeight 	= 2;	// hyperparameter, must be divisible by input widen
	config_3.kernelWidth  	= 2;	// hyperparameter, must be divisible by input width

	FlattenConfig config_4;
	config_4.layerType	= "flatten";
	config_4.inputHeight = 5;
	config_4.inputWidth = 5;
	config_4.inputDepth = 12;

	DenseConfig config_5;
	config_5.layerType = "dense";
	config_5.inputWidth = 300;
	config_5.outputWidth = 256;
	config_5.actFun = tanhAct;
	config_5.dactFun = dtanhAct;
	config_5.learningRate = learningRate;
	

	DenseConfig config_6;
	config_6.layerType = "dense";
	config_6.inputWidth = 256;
	config_6.outputWidth = 10;
	config_6.actFun = tanhAct;
	config_6.dactFun = dtanhAct;
	config_6.learningRate = learningRate;
	
	SoftmaxConfig config_7;
	config_7.layerType = "softmax";
	config_7.inputWidth = 10;
	config_7.outputWidth = 10;
	
	
	config.push_back(&config_0);
	config.push_back(&config_1);
	config.push_back(&config_2);
	config.push_back(&config_3);
	config.push_back(&config_4);
	config.push_back(&config_5);
	config.push_back(&config_6);
	config.push_back(&config_7);

	ConvolutionalNeuralNetwork * cnn == nullptr;

	/* - Training with loaded data */
	std::ofstream log("./log/RMSE.txt");
	log << "RMSE" << " " << "ACC" << '\n';
	std::cout << std::fixed << std::setprecision(4);
	for(int i = 0; i < epoch; i++){
		srand(time(NULL));
		/* Train and validate after each batch*/
        loadDataset(input_data, output_data, targetOutputs, labelVec, batchSize);
		Scalar LOSS = cnn ->train(input_data, output_data, batchSize);
		cleanDataBuffer(input_data);
		cleanLabelBuffer(output_data);
		loadTestData(input_data, output_data, targetOutputs, validateLabels, TEST);
		Scalar ACC = cnn ->validate(input_data, output_data, outputToLabelIdx, TEST);
		cleanDataBuffer(input_data);
		cleanLabelBuffer(output_data);

		/* Save to log file after each epoch */
		log << LOSS << " " << ACC << '\n';

	 	std::cout << "\rEpoch : " << i + 1 << " ACC: " << ACC << " LOSS: " << LOSS << std::endl;
	}
	std::cout << std::endl;
	log.close();

	/* - Calculate model accuracy */
	loadTestData(input_data, output_data, targetOutputs, validateLabels, TEST);
	Scalar accuracy = cnn ->validate(input_data, output_data, outputToLabelIdx, TEST);
	cleanDataBuffer(input_data);
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
		cnn ->propagateForward(test);							// propagate forward testing input matrix vector
		img.testing();										// display test sample
		
		int num = outputToLabelIdx(cnn ->layer.back() -> outputRef().back());
		std::cout << "Predicted output: " << num << " Confident: " << std::fixed << std::setprecision(2) << cnn ->layer.back() -> outputRef().back() -> coeff(num) << std::endl;
		std::cout << "Output vector: " << *cnn ->layer.back() -> outputRef().back() << std::endl;
		getline(std::cin, cmd);
	}
	return 0;
}

#else 
	
int main(int argc, char ** argv){
	srand(time(NULL));
	// loading dataset
	/* - Fist is setup input data containter, then the expected output data container */
	std::vector<std::vector<Matrix *>> input_data;
	std::vector<std::vector<Matrix *>> output_data;
	std::vector<std::vector<Matrix *>> input_test;
	std::vector<std::vector<Matrix *>> output_test;
	std::vector<std::vector<Matrix>>   targetOutputs(10, std::vector<Matrix>(1, Matrix(1, 10))); 
	
    /* - Create labeled array for mapping */
    std::vector<int*> labelVec;
    std::ifstream labelFile;
    labelFile.open("dataset/training_dataset/label.txt");
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
	
	// network training parameter
	int batchSize = 0;
	Scalar learningRate = 0.001;
	Scalar momentum = 0.9;
	Scalar decay_rate = 0.001;

	Optimizer * optimizer = nullptr;

	std::vector<LayerConfig *> config;
	
	// LeNet configuration
	ConvConfig config_0;
	config_0.layerType	  = "conv";
	config_0.inputHeight  = 28;
	config_0.inputWidth   = 28;
	config_0.inputDepth   = 1;
	config_0.kernelHeight = 5;	// hyperparameter
	config_0.kernelWidth  = 5;	// hyperparameter
	config_0.numKernel    = 6;		// hyperparameter
	config_0.padding      = 2;		// hyperparameter, usually zero
	config_0.striding     = 1;		// hyperparameter ?? (unsure, use default)
	config_0.actFun		  = LeakyReLU;
	config_0.dactFun	  = dLeakyReLU;
	config_0.opt 		  = optimizer;

	PoolingConfig config_1;
	config_1.layerType		= "maxpool";
	config_1.inputHeight  	= 28;	
	config_1.inputWidth   	= 28;	
	config_1.inputDepth   	= 6;	// depend on previous hyperparameter
	config_1.kernelHeight 	= 2;	// hyperparameter, must be divisible by input widen
	config_1.kernelWidth  	= 2;	// hyperparameter, must be divisible by input width

	ConvConfig config_2;
	config_2.layerType		= "conv";
	config_2.inputHeight  	= 14;	 
	config_2.inputWidth   	= 14;	
	config_2.inputDepth   	= 6;		// depend on previous hyperparameter
	config_2.kernelHeight 	= 5;	// hyperparameter
	config_2.kernelWidth  	= 5;	// hyperparameter
	config_2.numKernel    	= 16;		// hyperparameter
	config_2.padding      	= 0;		// hyperparameter, usually zero
	config_2.striding     	= 1;		// hyperparameter ??? (unsure, use default)
	config_2.actFun			= LeakyReLU;
	config_2.dactFun		= dLeakyReLU;
	config_2.opt 			= optimizer;

	PoolingConfig config_3;
	config_3.layerType		= "maxpool";
	config_3.inputHeight  	= 10;	
	config_3.inputWidth   	= 10;	
	config_3.inputDepth   	= 16;	// depend on previous hyperparameter
	config_3.kernelHeight 	= 2;	// hyperparameter, must be divisible by input widen
	config_3.kernelWidth  	= 2;	// hyperparameter, must be divisible by input width

	FlattenConfig config_4;
	config_4.layerType	= "flatten";
	config_4.inputHeight = 5;
	config_4.inputWidth = 5;
	config_4.inputDepth = 16;

	DenseConfig config_5;
	config_5.layerType = "dense";
	config_5.inputWidth = 400;
	config_5.outputWidth = 120;
	config_5.actFun = LeakyReLU;
	config_5.dactFun = dLeakyReLU;
	config_5.opt = optimizer;

	DenseConfig config_6;
	config_6.layerType = "dense";
	config_6.inputWidth = 120;
	config_6.outputWidth = 84;
	config_6.actFun = LeakyReLU;
	config_6.dactFun = dLeakyReLU;
	config_6.opt = optimizer;

	DenseConfig config_7;
	config_7.layerType = "dense";
	config_7.inputWidth = 84;
	config_7.outputWidth = 10;
	config_7.actFun = linear;
	config_7.dactFun = dlinear;
	config_7.opt = optimizer;

	SoftmaxConfig config_8;
	config_8.layerType = "softmax";
	config_8.inputWidth = 10;
	config_8.outputWidth = 10;
	
	config.push_back(&config_0);
	config.push_back(&config_1);
	config.push_back(&config_2);
	config.push_back(&config_3);
	config.push_back(&config_4);
	config.push_back(&config_5);
	config.push_back(&config_6);
	config.push_back(&config_7);
	config.push_back(&config_8);

	ConvolutionalNeuralNetwork * cnn = nullptr;
	
	std::cout << std::fixed << std::setprecision(4);
	loadDataset(input_data, output_data, targetOutputs, labelVec, DATASIZE);
	loadTestData(input_test, output_test, targetOutputs, validateLabels, TESTSIZE);
	
	std::string cmd = "";
	std::cout << "Command$ ";
	getline(std::cin, cmd);
	std::cout << std::endl;
	while(cmd != "exit"){
		if(cmd == "test") {
			if(cnn != nullptr) {
				// get test sample
				Image img("test-img/test.bmp", -1);		// open image for diesplay
				ImageData obj("test-img/test.bmp");		// open image for extract data
				img.setInvert(false);

				Matrix testData(img.getHeight(), img.getWidth()); 	// input data storage
				obj.getPixelMatrix(&testData);						// load pixel data into testing input matrix
				std::vector<Matrix *> test;							// create vector containter for input matrix
				test.push_back(&testData);							
				cnn ->propagateForward(test);							// propagate forward testing input matrix vector
				img.testing();										// display test sample
				
				int num = outputToLabelIdx(cnn ->layer.back() -> outputRef().back());
				std::cout << "Predicted output: " << num << " Confident: " << std::fixed << std::setprecision(2) << cnn ->layer.back() -> outputRef().back() -> coeff(num) << std::endl;
				std::cout << "Output vector: " << *cnn ->layer.back() -> outputRef().back() << std::endl;
			} else {
				std::cout << "No model for test" << std::endl;
			}
		} else if (cmd == "train") {
			if(cnn != nullptr){
				/* - Training with loaded data */
				std::string buffer;
				std::cout << "Epoch: ";
				getline(std::cin, buffer);
				int epoch = atoi(buffer.c_str());
				std::ofstream log("./log/RMSE.txt");
				log << "LOSS" << " " << "ACC" << '\n';
				std::cout << std::fixed << std::setprecision(4);
				for(int i = 0; i < epoch; i++){
					/* Train and validate after each batch*/
					Scalar LOSS = (cnn -> train(input_data, output_data, batchSize));
					Scalar ACC = (cnn -> validate(input_test, output_test, outputToLabelIdx, TESTSIZE));

					// update leanring rate

					std::cout << "\rEpoch : " << i + 1 << " Acc: " << ACC << " Lr: " << optimizer -> getLearningRate() << " Loss: " << LOSS << std::endl;
					optimizer -> ScheduleLearningRate(Scalar(i + 1));
					log << LOSS << " " << ACC << '\n';
				}
				std::cout << std::endl;
				log.close();
			} else {
				std::cout << "Not trainable network" << std::endl;
			}
		}
		else if (cmd == "new") {
			if(cnn != nullptr) cnn -> deleteNetwork();
			cnn = new ConvolutionalNeuralNetwork(config);
			std::string buffer;
	
			std::cout << "Learn rate: ";
			getline(std::cin, buffer);
			learningRate = atof(buffer.c_str());
			
			std::cout << "Momentum: ";
			getline(std::cin, buffer);
			momentum     = atof(buffer.c_str());
			
			std::cout << "Batch size: ";
			getline(std::cin, buffer);
			batchSize	    = atoi(buffer.c_str());
			
			std::cout << "Decay rate: ";
			getline(std::cin, buffer);
			decay_rate	= atof(buffer.c_str());
			
			optimizer = new SGD(learningRate, momentum, new ExponentDecayLearnRate(decay_rate));
			config_0.opt 		  = optimizer;
			config_2.opt 		  = optimizer;
			config_5.opt 		  = optimizer;
			config_6.opt 		  = optimizer;
			config_7.opt 		  = optimizer;
			cnn ->summary();
			
		} else if (cmd == "clean"){
			delete cnn;
			delete optimizer;
			cnn = nullptr;
			optimizer = nullptr;
		}else cmd = "";
		std::cout << "Command$ ";
		getline(std::cin, cmd);
		std::cout << std::endl;
	}
	cleanDataBuffer(input_data);
	cleanLabelBuffer(output_data);
	cleanDataBuffer(input_test);
	cleanLabelBuffer(output_test);
	while(config.size() != 0){
		config.pop_back();
	}
	// int epoch = atoi(argv[1]);
	// std::vector<std::vector<Matrix *>> input_set;
	// input_set.push_back(std::vector<Matrix *>());
	// input_set.push_back(std::vector<Matrix *>());
	// input_set.push_back(std::vector<Matrix *>());
	// input_set.push_back(std::vector<Matrix *>());

	// input_set[0].push_back(new Matrix(1, 2));
	// input_set[1].push_back(new Matrix(1, 2));
	// input_set[2].push_back(new Matrix(1, 2));
	// input_set[3].push_back(new Matrix(1, 2));

	// *input_set[0][0] << 0, 0;
	// *input_set[1][0] << 0, 1;
	// *input_set[2][0] << 1, 0;
	// *input_set[3][0] << 1, 1;

	// std::vector<std::vector<Matrix *>> output_set;
	// output_set.push_back(std::vector<Matrix *>());
	// output_set.push_back(std::vector<Matrix *>());
	// output_set.push_back(std::vector<Matrix *>());
	// output_set.push_back(std::vector<Matrix *>());

	// output_set[0].push_back(new Matrix(1, 1));
	// output_set[1].push_back(new Matrix(1, 1));
	// output_set[2].push_back(new Matrix(1, 1));
	// output_set[3].push_back(new Matrix(1, 1));

	// *output_set[0][0] << 0;
	// *output_set[1][0] << 1;
	// *output_set[2][0] << 1;
	// *output_set[3][0] << 0;

	// for(int i = 0; i < 4; i++);

	// DenseConfig config_in;
	// config_in.layerType = "dense";
	// config_in.inputWidth = 2;
	// config_in.outputWidth = 3;
	// config_in.actFun = ReLU;
	// config_in.dactFun = dReLU;
	// config_in.learningRate = 0.01;

	// DenseConfig config_0;
	// config_0.layerType = "dense";
	// config_0.inputWidth = 3;
	// config_0.outputWidth = 2;
	// config_0.actFun = ReLU;
	// config_0.dactFun = dReLU;
	// config_0.learningRate = 0.01;

	// DenseConfig config_out;
	// config_out.layerType = "dense";
	// config_out.inputWidth = 2;
	// config_out.outputWidth = 1;
	// config_out.actFun = ReLU;
	// config_out.dactFun = dReLU;
	// config_out.learningRate = 0.01;
	
	// std::vector<LayerConfig *> config;
	// config.push_back(&config_in);
	// config.push_back(&config_0);
	// config.push_back(&config_out);

	// ConvolutionalNeuralNetwork nn(config);

	// for(int i = 0; i < epoch; i++) {
	// 	nn.train(input_set, output_set, 4);
	// }

	// for(size_t i = 0; i < input_set.size(); i++) {
	// 	std::cout << "Input data " << *input_set[i][0] << ", ";
	// 	nn.propagateForward(input_set[i]);
	// 	for(size_t i = 0; i < nn.outputRef().size(); i++){
	// 		std::cout << "Output layer " << i << ": ";
	// 		std::cout << *nn.outputRef()[i] << std::endl;
	// 	}
	// }

	// while(input_set.size() != 0) {
	// 	while(input_set.back().size() != 0){
	// 		delete input_set.back().back();
	// 		input_set.back().pop_back();
	// 	}
	// 	input_set.pop_back();
	// }

	// while(output_set.size() != 0) {
	// 	while(output_set.back().size() != 0){
	// 		delete output_set.back().back();
	// 		output_set.back().pop_back();
	// 	}
	// 	output_set.pop_back();
	// }

	// softmax test => pass
	return 0;
}
#endif