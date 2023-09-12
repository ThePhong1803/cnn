#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <common.h>
#define alpha 0.01 // leaky relu param
#define SinP_coeff 2 // Sin polynomial coeff

// list of actiation function
Scalar Sigmoid(Scalar x);
Scalar dSigmoid(Scalar x);
Scalar ReLU(Scalar x);
Scalar dReLU(Scalar x);
Scalar LeakyReLU(Scalar x);
Scalar dLeakyReLU(Scalar x);
Scalar tanhAct(Scalar x);
Scalar dtanhAct(Scalar x);
Scalar SiLU(Scalar x);
Scalar dSiLU(Scalar x);
Scalar SinP(Scalar x);
Scalar dSinP(Scalar x);
#endif