#ifndef __LOSSCALC_H__
#define __LOSSCALC_H__

#include <common.h>
// Mean square error loss function
Scalar 				   MeanSquareError(Matrix * vec, Matrix * expected);
std::vector<Matrix *> dMeanSquareError(std::vector<Matrix *> * output, std::vector<Matrix *> * expected);

// Binary crosss entropy loss function
Scalar 				   BinaryCrossEntropy(Matrix * vec, Matrix * expected);
std::vector<Matrix *> dBinaryCrossEntropy(std::vector<Matrix *> * output, std::vector<Matrix *> * expected);

// Categorical crosss entropy loss function
Scalar 				   CategoricalCrossEntropy(Matrix * vec, Matrix * expected);
std::vector<Matrix *> dCategoricalCrossEntropy(std::vector<Matrix *> * output, std::vector<Matrix *> * expected);
#endif