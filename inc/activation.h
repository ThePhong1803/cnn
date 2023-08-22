#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <common.h>
#define alpha 0.01 // leaky rely param

// list of actiation function
Scalar Sigmoid(Scalar x);
Scalar dSigmoid(Scalar x);
Scalar ReLU(Scalar x);
Scalar dReLU(Scalar x);
Scalar LeakyReLU(Scalar x);
Scalar dLeakyReLU(Scalar x);
Scalar tanhAct(Scalar x);
Scalar dtanhAct(Scalar x);

#endif