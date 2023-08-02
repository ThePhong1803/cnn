#ifndef __COMMON_H__
#define __COMMON_H__
#endif

/* optimization preproces directive */
// #define EIGEN_MPL2_ONLY
// #define NDEBUG

/* Begin include section */
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Eigen>
#include <assert.h>
#include <iomanip>
#include <iomanip>
#include <activation.h>
#include <stdint.h>
/* End include section */

/* Begin typedef section */

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
typedef uint32_t uint;
typedef Scalar (*ScalarFunPtr)(Scalar);

/* End typedef section */