
#include <iostream>
#include <vector>
#include <iomanip>
#include <time.h>
#include <neuralnetwork.h>
#include <assert.h>
#include <convolutionalLayer.h>
#include <eigen3/Eigen/Eigen>
#include <imagedata.h>
#include <math.h>

int main() {
	Eigen::MatrixXf kernel(5,5);
	ImageData img("test-img/test.bmp");
	Matrix mat(img.height, img.width);
	img.getPixelMatrix(&mat);
	// for(int i = 0; i < mat.rows(); i++) {
	// 	for(int j = 0; j < mat.cols(); j++){
	// 		mat.coeffRef(i, j) = i * mat.cols() + j;
	// 	}
	// }
	
	// for(int i = 0; i < kernel.rows(); i++) {
	// 	for(int j = 0; j < kernel.cols(); j++){
	// 		kernel.coeffRef(i, j) = i * kernel.cols() + j;
	// 	}
	// }
	kernel << -1, -1, -1, 1, 1,
			  -1, -1, 1, 1, 1,
			  -1, 1, 1, 1, -1,
			  1, 1, 1, -1, -1,
			  1, 1, -1, -1, -1;
	std::cout << "Input matrix (A): " << std::endl;
	std::cout << mat << std::endl;
	std::cout << "Kernel matrix (K): " << std::endl;
	std::cout << kernel << std::endl;
	Matrix res = conv(mat, kernel);
	std::cout << "Res: " << std::endl;
	std::cout << res << std::endl;
	char map[] = " .:-=+*#%@";
	int idx = 0;
	for(int i = 0; i < res.rows(); i++){
		for(int j = 0; j < res.cols(); j++){
			idx = (int)res.coeff(i, j) % 10;
			std::cout << map[idx];
		}
		std::cout << std::endl;
	}
	return 0;
}