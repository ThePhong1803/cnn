#include "image.h"
#include <eigen3/Eigen/Eigen>
#include <assert.h>
#pragma once

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
typedef uint32_t uint;


class ImageData {
	public:
	// data wrapper attribute
	int number;
	int id;
	int height;
	int width;
	std::vector<float> pixels;
	
	// constructor and destructor
	ImageData();
	ImageData(std::vector<float> _pixels, int _number = 0, int _id = 0);
	ImageData(std::string path, bool invert = false, int _number = 0, int _id = 0);
	~ImageData();
	
	// data wrapper methoid
	void getPixelVector(RowVector * temp);
	void getPixelMatrix(Matrix * mat);
};