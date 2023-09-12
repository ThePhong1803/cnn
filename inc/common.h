#ifndef __COMMON_H__
#define __COMMON_H__

/* optimization preproces directive */
#define EIGEN_MPL2_ONLY
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
#include <stdint.h>
#include <algorithm>
#include <iterator>
#include <random>
/* End include section */

/* Begin typedef section */

typedef double Scalar;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::RowVectorXd RowVector;
typedef Eigen::VectorXd ColVector;
typedef Scalar (*ScalarFunPtr)(Scalar);

/* End typedef section */

#endif