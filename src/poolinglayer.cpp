#include <poolinglayer.h>

// Pooling layer config implementation
PoolingConfig::PoolingConfig() : LayerConfig()
{
	// do nothing
}

PoolingConfig::~PoolingConfig()
{
	// do nothing
}

uint32_t &PoolingConfig::inputHeightRef()
{
	return this -> inputHeight;
}

uint32_t &PoolingConfig::inputWidthRef()
{
	return this -> inputWidth;
}

uint32_t &PoolingConfig::inputDepthRef()
{
	return this -> inputDepth;
}

uint32_t &PoolingConfig::kernelHeightRef()
{
	return this -> kernelHeight;
}

uint32_t &PoolingConfig::kernelWidthRef()
{
	return this -> kernelWidth;
}

// Utilities function
ElementPooling<Scalar> MaxPooling(Matrix * mat)
{
	// this function take in a matrix and return the index 
	// of the max element
	ElementPooling<Scalar> ret(0, 0, mat -> coeff(0, 0));
	for(int r = 0; r < mat -> rows(); r++)
	{
		for(int c = 0; c < mat -> cols(); c++ )
		{
			if(mat -> coeff(r, c) > ret.value){
				ret.value = mat -> coeff(r, c);
				ret.row = r;
				ret.col = c;
			}
		}
	}
	return ret;
}

// MaxPoolingLayer implementation
// Layer constructor implementation
MaxPoolingLayer::MaxPoolingLayer(PoolingConfig * _config) : Layer(), config(_config)
{
	/* Implement MaxPoolingLayer constructor */
	
	// calculate output matrix size from config
	// for this implementation, we will assume that the filter kernel
	// is divisible by the input matrix
	assert(config -> inputHeight % config -> kernelHeight == 0);
	assert(config -> inputWidth  % config -> kernelWidth == 0);
	uint32_t R = config -> inputHeight / config -> kernelHeight;
	uint32_t C = config -> inputWidth  / config -> kernelWidth;
	
	// create container for kernel matrix, only 1 kernel filter is needed
	for(uint32_t i = 0; i < config -> inputDepth; i++)
	{
		// create output layer and kernel layer
		kernel.push_back(new Matrix(config -> inputHeight, config -> inputWidth));
		output.push_back(new Matrix(R, C));
		kernel.back() -> setZero();
		output.back() -> setZero();
	}
}

// Layer destructor implementation
MaxPoolingLayer::~MaxPoolingLayer()
{
	/* Implement MaxPoolingLayer destructor */
	// delete this layer containter for pooled index
	while(kernel.size() != 0)
	{
		delete kernel.back();
		kernel.pop_back();
	}
	
	// delete this layer container for output
	while(output.size() != 0)
	{
		delete output.back();
		output.pop_back();
	}
}

// Input and output access for base claas pointer
std::vector<Matrix *> &MaxPoolingLayer::inputRef()
{
	return this -> input;
}

std::vector<Matrix *> &MaxPoolingLayer::outputRef()
{
	return this -> output;
}

void MaxPoolingLayer::propagateForward(std::vector<Matrix*> * input)
{
	// perform pooling for the max value to output layer,
	// also store the indecies of the max value
	for(size_t l = 0; l < output.size(); l++){
		// loop throught all output layer
		for(int r = 0; r < output[l] -> rows(); r++)
		{
			for(int c = 0; c < output[l] -> cols(); c++)
			{
				// select max element from sub matrix
				// calculate current offset for current pooling chunk
				uint32_t rows_offset = r * config -> kernelHeight;
				uint32_t cols_offset = c * config -> kernelWidth;
				// create a temp matrix for currently pooled block				
				Matrix temp = (*input)[l] -> block(rows_offset, cols_offset, config -> kernelHeight, config -> kernelWidth);
				// Seach the max element in current block and return its local indecies
				ElementPooling<Scalar> max = MaxPooling(&temp);
				// Pass max element to output and marking the absolute indecies of pooled element
				output[l] -> coeffRef(r, c) = max.value;
				kernel[l] -> coeffRef(max.row + rows_offset, max.col + cols_offset) = 1;
			}
		}
	}
}

void MaxPoolingLayer::propagateBackward(std::vector<Matrix*> * errors)
{
	// create storage for new error matrix
	std::vector<Matrix *> temp;
	for(size_t i = 0; i < config -> inputDepth; i++)
	{
		temp.push_back(new Matrix(config -> inputWidth, config -> inputWidth));
	}
	for(size_t l = 0; l < errors -> size(); l++){
		// loop throught all output layer
		for(int r = 0; r < (*errors)[l] -> rows(); r++)
		{
			for(int c = 0; c < (*errors)[l] -> cols(); c++)
			{
				// select max element from sub matrix
				// calculate current offset for current pooling chunk
				uint32_t rows_offset = r * config -> kernelHeight;
				uint32_t cols_offset = c * config -> kernelWidth;
				// take out kernel part			
				temp[l] -> block(rows_offset, cols_offset, config -> kernelHeight, config -> kernelWidth) = (*errors)[l] -> coeff(r, c) * kernel[l] -> block(rows_offset, cols_offset, config -> kernelHeight, config -> kernelWidth);
			}
		}
	}
	
	// empty errros matrix vector
	// tranfer from temp error matrix vector to error matrix vector
	for(size_t l = 0; l < config -> inputDepth; l++){
		delete (*errors)[l];
		(*errors)[l] = temp[l];
		temp.pop_back();

	}
}