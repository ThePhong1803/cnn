#include <optimizer.h>

/* Base class */
Optimizer::Optimizer()
{
    // Do nothing
}

Optimizer::~Optimizer()
{
    // Base class virutal destructor
}

/* Base class method is not callable */
void Optimizer::DenseOptimizer(DenseLayer * layer, int batch_size)
{
    throw DisableMethod("DenseOptimizer");
}
void Optimizer::ConvOptimizer(ConvolutionalLayer * layer, int batch_size)
{
    throw DisableMethod("ConvOptimizer");
}

/* SDG method implementation */
SGD::SGD() : Optimizer() 
{
    // do nothing
}

SGD::~SGD() {
    // do nothing
};
/* These function should be call when the layer in network perform weights and biases update after one batch*/
void SGD::DenseOptimizer(DenseLayer * layer, int batch_size)
{
    // TODO: Implement SGD Optimization for Dense Layer
    *(layer -> weight) += (*(layer -> dweight) / Scalar(batch_size));
    *(layer -> biases) += (*(layer -> dbiases) / Scalar(batch_size));

    // reset change in weights and biases
    layer -> dweight -> setZero();
    layer -> dbiases -> setZero();

}
void SGD::ConvOptimizer(ConvolutionalLayer * layer, int batch_size)
{
    // TODO: Implement SGD Optimization for Convolutional Layer
}