#include <activation.h>

// activation function 
Scalar Sigmoid(Scalar x)
{
	// sigmoid
	return 1.0f/(1 + exp(-x));
}

Scalar dSigmoid(Scalar x)
{
	// sigmiod derivative
	return Sigmoid(x) * (1 - Sigmoid(x));
}

Scalar ReLU(Scalar x)
{
	// ReLU
	return (x >= 0) ?  x : 0;
}

Scalar dReLU(Scalar x)
{
	// ReLU derivative
	return (x >= 0) ?  1 : 0;
}

Scalar LeakyReLU(Scalar x)
{
	// LeakyReLU
	return (x > 0) ?  x : alpha * x;
}

Scalar dLeakyReLU(Scalar x)
{
	// leakyReLU derivative
	return (x >= 0) ?  1 : alpha;
}

Scalar tanhAct(Scalar x)
{
	return std::tanh(x);
}

Scalar dtanhAct(Scalar x)
{
	return 1 - std::tanh(x) * std::tanh(x);
}