#include <optimizer.h>


/* Base class method is not callable */
void Optimizer::DenseOptimizer(DenseLayer * layer)
{
    throw DisableMethod("DenseOptimizer");
}
void Optimizer::ConvOptimizer(ConvolutionalLayer * layer)
{
    throw DisableMethod("ConvOptimizer");
}

/* SDG method implementation */
/* These function should be call when the layer in network perform weights and biases update after one batch*/
void SDG::DenseOptimizer(DenseLayer * layer)
{
    // TODO: Implement SGD Optimization for Dense Layer

}
void SDG::ConvOptimizer(ConvolutionalLayer * layer)
{
    // TODO: Implement SGD Optimization for Convolutional Layer
}