#include <optimizer.h>

/* SDG method implementation */
SGD::SGD()
{
	// default optimizer contructor
	this -> learnRate 	= 0.01f;
	this -> momentum 	= 0.90f;
}

SGD::SGD(Scalar _learnRate, Scalar _momentum) : Optimizer(), learnRate(_learnRate), momentum(_momentum)
{
    // disable nestorov
}	

SGD::~SGD() {
    // do nothing
};
/* These function should be call when the layer in network perform weights and biases update after one batch*/
void SGD::DenseOptimizer(DenseLayer * layer, int batch_size)
{
    // TODO: Implement SGD Optimization for Dense Layer
    *(layer -> dweight) /= Scalar(batch_size);
    *(layer -> dbiases) /= Scalar(batch_size);
	
	// calculate velocity
	(*layer -> vweight) = momentum * (*layer -> vweight) + learnRate * (*layer -> dweight);
	(*layer -> vbiases) = momentum * (*layer -> vbiases) + learnRate * (*layer -> dbiases);
	
	// weight update
	(*layer -> weight) += (*layer -> vweight);
	(*layer -> biases) += (*layer -> vbiases);

    // reset change in weights and biases
    layer -> dweight -> setZero();
    layer -> dbiases -> setZero();

}
void SGD::ConvOptimizer(ConvolutionalLayer * layer, int batch_size)
{
    // TODO: Implement SGD Optimization for Convolutional Layer
}