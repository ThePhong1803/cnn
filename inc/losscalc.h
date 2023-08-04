#ifndef __LOSSCALC_H__
#define __LOSSCALC_H__

#include <common.h>
// Mean square error loss function
Scalar 				   MeanSquareError(RowVector * vec, RowVector * expected);
std::vector<Matrix *> dMeanSquareError(std::vector<Matrix *> * output, std::vector<Matrix *> * expected);

// Binary crosss entropy loss function
Scalar 				   BinaryCrossEntropy(RowVector * vec, RowVector * expected);
std::vector<Matrix *> dBinaryCrossEntropy(std::vector<Matrix *> * output, std::vector<Matrix *> * expected);
#endif