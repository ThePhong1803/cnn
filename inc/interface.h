#ifndef __LAYER_H__
#define __LAYER_H__
#include <eigen3/Eigen/Eigen>
#include <vector>
#include <interface.h>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
typedef uint32_t uint;

/* Create a interface for different type of layer int network */
class Layer {
	public:
	Layer() {
		std::cout << "Layer constructor" << std::endl;
	}
	virtual ~Layer() {
		std::cout << "Layer destructor" << std::endl;
	}
	virtual void propagateForward(std::vector<Matrix*> &input) = 0;
	virtual void propagateBackward(std::vector<Matrix*> &output) = 0;
};
#endif