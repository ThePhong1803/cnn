#include <optimizer.h>

/* SDG method implementation */
SGD::SGD()
{
	// default optimizer contructor
	this -> learnRate 	    = 0.01f;
	this -> momentum 	    = 0.90f;
    this -> lr_scheduler    = new LearningRateScheduler();
}

SGD::SGD(Scalar _learnRate, Scalar _momentum) : Optimizer(), learnRate(_learnRate), momentum(_momentum)
{
    this -> lr_scheduler    = new LearningRateScheduler(_learnRate);
}	

SGD::SGD(Scalar _learnRate, Scalar _momentum, LearningRateScheduler * _lr_scheduler) : Optimizer(), learnRate(_learnRate), momentum(_momentum)
{
    this -> lr_scheduler = _lr_scheduler;
    this -> lr_scheduler -> lr = _learnRate;
}	

SGD::~SGD() {
    // do nothing
    delete this -> lr_scheduler;
};
/* These function should be call when the layer in network perform weights and biases update after one batch*/
void SGD::DenseOptimizer(DenseLayer * layer, int batch_size)
{
    // Implement SGD Optimization for Dense Layer
    *(layer -> dweight) /= Scalar(batch_size);
    *(layer -> dbiases) /= Scalar(batch_size);
	
	// calculate velocity
	(*layer -> vweight) = momentum * (*layer -> vweight) + (*layer -> dweight) * (1 - momentum);
	(*layer -> vbiases) = momentum * (*layer -> vbiases) + (*layer -> dbiases) * (1 - momentum);
	
	// weight update
	(*layer -> weight) -= learnRate * (*layer -> vweight);
	(*layer -> biases) -= learnRate * (*layer -> vbiases);

    // reset change in weights and biases
    layer -> dweight -> setZero();
    layer -> dbiases -> setZero();

}
void SGD::ConvOptimizer(ConvolutionalLayer * layer, int batch_size)
{
    // Implement SGD Optimization for Convolutional Layer
    for(size_t i = 0; i < layer -> kernel.size(); i++)
	{
		// std::cout << layer -> config -> layerType << " kernel " << i << std::endl;
		//loop through all kernel in kernel vector
		for(size_t l = 0; l < layer -> kernel[i].size(); l++)
		{
			//Loop through all kernel layer in kernel
			*layer -> dkernel[i][l] /= Scalar(batch_size);   // calc average gradient
            *layer -> vkernel[i][l] = momentum * (*layer -> vkernel[i][l]) + (*layer -> dkernel[i][l]) * (1 - momentum);
            *layer -> kernel[i][l] -= learnRate * (*layer -> vkernel[i][l]);
			
			// std::cout << "Layer " << l << std::endl;
			// std::cout << *layer -> kernel[i][l] << std::endl;

            // reset kernel layer gradient
            layer -> dkernel[i][l] -> setZero();
 		}
	}

    // loop through all biases matrix in bias matrix vector
    for(size_t i = 0; i < layer -> biases.size(); i++)
    {
		// std::cout << layer -> config -> layerType << " biases " << i << std::endl;
        *layer -> dbiases[i] /= Scalar(batch_size);
        *layer -> vbiases[i] = momentum * (*layer -> vbiases[i]) + (*layer -> dbiases[i]) * learnRate * (1 - momentum);
        *layer -> biases[i] -= learnRate * (*layer -> vbiases[i]);
		
		// std::cout << *layer -> biases[i] << std::endl;
		
        // reset biases matrix gradient
        layer -> dbiases[i] -> setZero();
    }
}

Scalar SGD::getLearningRate(){
    return this -> learnRate;
}

void SGD::ScheduleLearningRate(Scalar step) {
    this -> lr_scheduler -> updateLearningRate(step);
    this -> learnRate = lr_scheduler -> getLearningRate();
}

// Learning rate scheduling section
ExponentDecayLearnRate::ExponentDecayLearnRate() : LearningRateScheduler()
{
    this -> decay_factor = 0.1f;
}

ExponentDecayLearnRate::ExponentDecayLearnRate(Scalar _decay_factor) : LearningRateScheduler(), decay_factor(_decay_factor)
{
    // Do nothing here
}

ExponentDecayLearnRate::ExponentDecayLearnRate(Scalar _learnRate, Scalar _decay_factor) : LearningRateScheduler(_learnRate), decay_factor(_decay_factor)
{
    // Do nothing here
}

ExponentDecayLearnRate::~ExponentDecayLearnRate()
{
    // Deconstructor
}

void ExponentDecayLearnRate::updateLearningRate(Scalar step)
{
    this -> lr = this -> lr * exp(-decay_factor * Scalar(step));
}