#include <convolutionalLayer.h>

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
	int resRows = (input.rows() - kernel.rows()) / striding + 1;
	int resCols = (input.cols() - kernel.cols()) / striding + 1;
	Matrix res(resRows, resCols);
	
	// loop throught each element of the output matrix and calculate base on the input matrix and input kernel matrix
	for(int i = 0; i < res.rows(); i++){
		for(int j= 0; j < res.cols(); j++) {
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
	int resRows = (input.rows() - rot180Kernel.rows()) / striding + 1;
	int resCols = (input.cols() - rot180Kernel.cols()) / striding + 1;
	Matrix res(resRows, resCols);
	
	// loop throught each element of the output matrix and calculate base on the input matrix and input kernel matrix
	for(int i = 0; i < res.rows(); i++){
		for(int j= 0; j < res.cols(); j++) {
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


ConvolutionalLayer::ConvolutionalLayer(LayerConfig * _config) : Layer(), config(_config) {
	/* Implement ConvolutionalLayer constructor 
		The toplogoy input should be: {h, w, d, m, n, p, k}
		- h, w, d is the input image height, width, depth respectively
		- m, n, p is the size of kernel height, width, depth respectively
	    - k is the number of kernel in this layer
	*/
	// extract layer configuration
	// int h 			= this -> config -> imageHeight;		// input image height
	// int w 			= this -> config -> imageWidth;			// input image width
	// int d 			= this -> config -> imageDepth;			// input image depth
	// int m 			= this -> config -> kernelHeight;		// kernel height
	// int n 			= this -> config -> kernelWidth;		// kernel width
	// int p 			= this -> config -> kernelDepth;		// kernel depth
	// int k 			= this -> config -> numKernel;			// number of filter kernel
	// int padding		= this -> config -> padding;			// padding adding to input image
	// int striding 	= this -> config -> striding;			// striding for correleation operation
	
	// // calculate output matrix size
	// int R = (h - m) / striding + 1;
	// int C = (w - n) / striding + 1;
	
	// // create container for output, and bias matrix
	// for(int i = 0; i < k; i++){
		// output.push_back(new Matrix(R, C));
		// bias.push_back(new Matrix(R, C));
	// }
	std::cout << "ConvolutionalLayer constructor" << std::endl;
}

ConvolutionalLayer::~ConvolutionalLayer(){
	/* Implement ConvolutionalLayer destructor */
	std::cout << "ConvolutionalLayer destructor" << std::endl;
}

// this function perform network forward propagation
void ConvolutionalLayer::propagateForward(std::vector<Matrix> &input) {}

// thsi frunction perform network backward propagateion
void ConvolutionalLayer::propagateBackward(std::vector<Matrix> &output) {}