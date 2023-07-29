#include <convolutionallayer.h>

// Correlation and Convolution operation implementation

Matrix corr(Matrix &mat, Matrix &kernel, uint32_t padding, uint32_t striding) {
	/* 
		This function perform correlation operation between input matrix and kernel matrix
		and return the result;
		@param: mat is the input matrix
		@param: kernel is the kernel matrix for correlation operation
		@param: padding is the additinal number of row and collum added to input matrix
		@param: striding the the step for kernel filter to slide over the input matrix
	*/

	// conditoin for the correlation operation
	assert(striding > 0 && mat.rows() + 2 * padding >= kernel.rows() && mat.cols() + 2 * padding >= kernel.cols());
	
	/*
		Create a temporary input matrix base on input matrix with padding added
		for a nxn input matrix with p padding create a (n + 2p) x (n + 2p) matrix
		and initialize the temporary input matrix with zero
	*/
	Matrix input(mat.rows() + 2 * padding, mat.cols() + 2 * padding);
	input.setZero();
	
	/* 
		perform block copy the original input matrix to new resized input matrix;
		start from index = padding, and copy the entire input matrix into temp matrix;
	*/
	input.block(padding, padding, mat.rows(), mat.cols()) = mat;
	
	/* 
		create container for output matrix of the correlation operation, 
		the matrix size is [(n + 2p -k)/s] + 1
			n: input matrix size (n x n)
			p: padding added to input matrix
			k: kernel matrix size
			s: striding value 
		
		int our case, the formula would be:	[(w - k)/s] + 1, with:
			w: temp input matrix size (w x w) or (n + 2p) x (n + 2p)
	*/
	uint32_t resRows = (input.rows() - kernel.rows()) / striding + 1;
	uint32_t resCols = (input.cols() - kernel.cols()) / striding + 1;
	Matrix res(resRows, resCols);
	
	// loop throught each element of the output matrix and calculate base on the input matrix and input kernel matrix
	for(uint32_t i = 0; i < res.rows(); i++)
	{
		for(uint32_t j= 0; j < res.cols(); j++) 
		{
			/*
				Start from the first row, col of input matrix, each time take out 1 block which size is 
				equal to kernel matrix, for sitriding, we decrease striding value by 1 because when calculate 
				the output matrix using its indexes, they increase by 1 each loop. By multiplying them by striding
				result in aligining them correctly
			*/
			res.coeffRef(i, j) = (input.block(i * striding, j * striding, kernel.rows(), kernel.cols()).array() * kernel.array()).sum();
		}
	}
	return res;
}

Matrix conv(Matrix &mat, Matrix &kernel, uint32_t padding, uint32_t striding){
	// the same as correlation but with 180-degree rotated kernel
	Matrix rot180Kernel = kernel.reverse();
		/* 
		This function perform convolution operation between input matrix and kernel matrix
		and return the result;
		@param: mat is the input matrix
		@param: kernel is the kernel matrix for correlation operation
		@param: padding is the additinal number of row and collum added to input matrix
		@param: striding the the step for kernel filter to slide over the input matrix
	*/

	// conditoin for the correlation operation
	assert(striding > 0 && mat.rows() + 2 * padding >= rot180Kernel.rows() && mat.cols() + 2 * padding >= rot180Kernel.cols());
	
	/*
		Create a temporary input matrix base on input matrix with padding added
		for a nxn input matrix with p padding create a (n + 2p) x (n + 2p) matrix
		and initialize the temporary input matrix with zero
	*/
	Matrix input(mat.rows() + 2 * padding, mat.cols() + 2 * padding);
	input.setZero();
	
	/* 
		perform block copy the original input matrix to new resized input matrix;
		start from index = padding, and copy the entire input matrix into temp matrix;
	*/
	input.block(padding, padding, mat.rows(), mat.cols()) = mat;
	
	/* 
		create container for output matrix of the correlation operation, 
		the matrix size is [(n + 2p -k)/s] + 1
			n: input matrix size (n x n)
			p: padding added to input matrix
			k: kernel matrix size
			s: striding value 
		
		int our case, the formula would be:	[(w - k)/s] + 1, with:
			w: temp input matrix size (w x w) or (n + 2p) x (n + 2p)
	*/
	uint32_t resRows = (input.rows() - rot180Kernel.rows()) / striding + 1;
	uint32_t resCols = (input.cols() - rot180Kernel.cols()) / striding + 1;
	Matrix res(resRows, resCols);
	
	// loop throught each element of the output matrix and calculate base on the input matrix and input kernel matrix
	for(uint32_t i = 0; i < res.rows(); i++)
	{
		for(uint32_t j= 0; j < res.cols(); j++) 
		{
			/*
				Start from the first row, col of input matrix, each time take out 1 block which size is 
				equal to kernel matrix, for sitriding, we decrease striding value by 1 because when calculate 
				the output matrix using its indexes, they increase by 1 each loop. By multiplying them by striding
				result in aliginning them correctly
			*/
			res.coeffRef(i, j) = (input.block(i * striding, j * striding, rot180Kernel.rows(), rot180Kernel.cols()).array() * rot180Kernel.array()).sum();
		}
	}
	return res;
}


// ConvolutionalLayer config classs implementation
ConvConfig::ConvConfig() : LayerConfig()
{
	// do nothing
}

ConvConfig::~ConvConfig()
{
	// do nothing
}

uint32_t &ConvConfig::inputHeightRef()
{
	return this -> inputHeight;
}

uint32_t &ConvConfig::inputWidthRef()
{
	return this -> inputWidth;
}

uint32_t &ConvConfig::inputDepthRef()
{
	return this -> inputDepth;
}

uint32_t &ConvConfig::kernelHeightRef()
{
	return this -> kernelHeight;
}

uint32_t &ConvConfig::kernelWidthRef()
{
	return this -> kernelWidth;
}

uint32_t &ConvConfig::kernelDepthRef()
{
	// kernel depth is equal to current layer input depth
	return this -> inputDepth;
}

ScalarFunPtr &ConvConfig::activationFunctionRef()
{
	return this -> actFun;
}
ScalarFunPtr &ConvConfig::activationFunctionDerivativeRef()
{
	return this -> dactFun;
}

// ConvolutionalLayer implementation
ConvolutionalLayer::ConvolutionalLayer(ConvConfig * _config) : Layer(), config(_config) 
{
	// check condition for layer initialization
	/* Implement ConvolutionalLayer constructor */
	// check if config is valid (output is cover all input)
	assert((config -> inputHeight + 2 * config -> padding - config -> kernelHeight + 1) % config -> striding == 0);
	assert((config -> inputWidth  + 2 * config -> padding - config -> kernelWidth  + 1) % config -> striding == 0);

	// calculate output matrix size from config
	uint32_t R = (config -> inputHeight + 2 * config -> padding - config -> kernelHeight) / config -> striding + 1;
	uint32_t C = (config -> inputWidth  + 2 * config -> padding - config -> kernelWidth ) / config -> striding + 1;

	// create container for kernel matrix
	for(uint32_t i = 0; i < config -> numKernel; i++)
	{
		// adding kernel layer container
		kernel.push_back(std::vector<Matrix*>());
		for(uint32_t j = 0; j < config -> inputDepth; j++)
		{
			// add new kernel matrix layer to current kernel
			kernel.back().push_back(new Matrix(config -> kernelHeight, config -> kernelWidth));
			// init kernel with random value
			kernel.back().back() -> setRandom();
		}
	}

	// create container for output, and bias matrix
	for(uint32_t i = 0; i < config -> numKernel; i++)
	{
		// output, caches and biases with k layers
		output.push_back(new Matrix(R, C));
		caches.push_back(new Matrix(R, C));
		biases.push_back(new Matrix(R, C));
		
		// init output and bias with zero and random (zero for now)
		output.back() -> setZero();
		caches.back() -> setZero();
		biases.back() -> setZero();
	}
}

ConvolutionalLayer::~ConvolutionalLayer(){
	/* Implement ConvolutionalLayer destructor */
	// deallocate used mem in k kernel, k output layer and k biases
	while(kernel.size() != 0)
	{
		// deallocate used mem for kernel
		while(kernel.back().size() != 0)
		{
			// delete kernel layer mem
			delete kernel.back().back();
			// delete entry in kernel layers vector
			kernel.back().pop_back();
		}
		// delete kernel entry in kernel vector
		kernel.pop_back();
		
		// delete output, caches and biases mem
		delete output.back();
		delete caches.back();
		delete biases.back();
		
		// delete entry in output, caches and biases vectors
		output.pop_back();
		caches.pop_back();
		biases.pop_back();
	}
}

// function to access this layer input and output
std::vector<Matrix *> &ConvolutionalLayer::inputRef()
{
	return this -> input;
} 

std::vector<Matrix *> &ConvolutionalLayer::outputRef()
{
	return this -> output;
} 


// this function perform network forward propagation
void ConvolutionalLayer::propagateForward(std::vector<Matrix *> * input) {
	// for each filter kernel in kernel vector
	for(size_t i = 0; i < kernel.size(); i++)
	{
		// calculate correlation in each layer and add to caches layer
		// clean caches before calculation
		caches[i] -> setZero();
		for(size_t l = 0; l < kernel[i].size(); l++)
		{
			(*caches[i]) += corr(*(*input)[l], *kernel[i][l], config -> padding, config -> striding);
		}
		// adding biases to caches layer
		(*caches[i]) += (*biases[i]);
		// apply activation function in each output
		(*output[i]) = caches[i] -> unaryExpr(std::ptr_fun(config -> actFun));
	}
}

// this frunction perform network backward propagateion
void ConvolutionalLayer::propagateBackward(std::vector<Matrix *> * errors)
{
	// calculate layer error 
	std::vector<Matrix> delta(output.size());
	for(size_t i = 0; i < delta.size(); i++)
	{
		// loop throught all errors layer calculate layer errors
		Matrix d = ((*errors)[i] -> array()) * (caches[i] -> unaryExpr(std::ptr_fun(config -> dactFun)).array());
		// Matrix d = *(*errors)[i];
		delta[i] = d;	
	}

	// delete current error vector
	while(errors -> size() != 0)
	{
		// delete all errors layer util the errors vector is empty
		delete errors -> back();
		errors -> pop_back();
	}

	// resize error vecotr and errors layer matrix for previous layer

	// loop through each layer in input layer
	for(size_t l = 0; l < config -> inputDepth; l++)
	{
		// create error matrix for prev layer init with zeros
		errors -> push_back(new Matrix(config -> inputHeight, config -> inputWidth));
		errors -> back() -> setZero();
		// loop through all kernel in our kernel vector
		for(size_t k = 0; k < kernel.size(); k++)
		{
			// update kernel layer l with the calculated error
			(*kernel[k][l]) += config -> learningRate * corr(*input[l], delta[k]);
			// update errors vector for next layer
			(*(*errors)[l]) += conv(delta[k], *kernel[k][l], config -> kernelHeight - 1, 1);
			// update biases matrix
			(*biases[k]) += config -> learningRate * (delta[k]);
		}
	}
}