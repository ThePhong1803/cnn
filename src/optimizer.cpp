#include <optimizer.h>

/* SDG method implementation */
SGD::SGD()
{
	// default optimizer contructor
	this -> learnRate 	    = 0.001f;
	this -> momentum 	    = 0.0f;
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
	(*layer -> vweight) = momentum * (*layer -> vweight) + (- *layer -> dweight) * (1 - momentum);
	(*layer -> vbiases) = momentum * (*layer -> vbiases) + (- *layer -> dbiases) * (1 - momentum);

    // (*layer -> vweight) = (*layer -> dweight);
	// (*layer -> vbiases) = (*layer -> dbiases);
	
	// weight and biases update
	(*layer -> weight) += learnRate * (*layer -> vweight);
	(*layer -> biases) += learnRate * (*layer -> vbiases);

    // reset change in weights and biases
    layer -> dweight -> setZero();
    layer -> dbiases -> setZero();

}
void SGD::ConvOptimizer(ConvolutionalLayer * layer, int batch_size)
{
    // Implement SGD Optimization for Convolutional Layer
    for(size_t i = 0; i < layer -> kernel.size(); i++)
	{
		//loop through all kernel in kernel vector
		for(size_t l = 0; l < layer -> kernel[i].size(); l++)
		{
			//Loop through all kernel layer in kernel
			*layer -> dkernel[i][l] /= Scalar(batch_size);   // calc average gradient
            *layer -> vkernel[i][l] = momentum * (*layer -> vkernel[i][l]) + (- *layer -> dkernel[i][l]) * (1 - momentum);
            // *layer -> vkernel[i][l] = (*layer -> dkernel[i][l]);
            *layer -> kernel[i][l] += learnRate * (*layer -> vkernel[i][l]);

            // reset kernel layer gradient
            layer -> dkernel[i][l] -> setZero();
 		}
	}

    // loop through all biases matrix in bias matrix vector
    for(size_t i = 0; i < layer -> biases.size(); i++)
    {
		// std::cout << layer -> config -> layerType << " biases " << i << std::endl;
        *layer -> dbiases[i] /= Scalar(batch_size);
        *layer -> vbiases[i] = momentum * (*layer -> vbiases[i]) + (- *layer -> dbiases[i]) * (1 - momentum);
        // *layer -> vbiases[i] = (*layer -> dbiases[i]);
        *layer -> biases[i] += learnRate * (*layer -> vbiases[i]);

        // reset biases matrix gradient
        layer -> dbiases[i] -> setZero();
    }
}

Scalar SGD::getLearningRate(){
	this -> learnRate = this -> lr_scheduler -> getLearningRate();
    return this -> learnRate;
}

void SGD::setLearningRate(Scalar _new_lr){
	this -> lr_scheduler -> resetStep();
	this -> lr_scheduler -> setLearningRate(_new_lr);
	this -> learnRate = _new_lr;
}

void SGD::ScheduleLearningRate() {
    this -> lr_scheduler -> updateLearningRate();
    this -> learnRate = lr_scheduler -> getLearningRate();
}

// Learning rate scheduling section
ExponentDecayLearnRate::ExponentDecayLearnRate() : LearningRateScheduler()
{
	this -> lr_cnt = 0.0f;
    this -> decay_factor = 0.1f;
}

ExponentDecayLearnRate::ExponentDecayLearnRate(Scalar _decay_factor) : LearningRateScheduler(), decay_factor(_decay_factor)
{
    this -> lr_cnt = 0.0f;
}

ExponentDecayLearnRate::ExponentDecayLearnRate(Scalar _learnRate, Scalar _decay_factor) : LearningRateScheduler(_learnRate), decay_factor(_decay_factor)
{
    this -> lr_cnt = 0.0f;
}

ExponentDecayLearnRate::~ExponentDecayLearnRate()
{
    // Deconstructor
}

void ExponentDecayLearnRate::updateLearningRate()
{
	this -> lr_cnt += 1.0f;
    this -> lr = this -> lr * decay_factor;
}

void ExponentDecayLearnRate::resetStep()
{
    this -> lr_cnt = 0;
}

// Learning rate scheduling section
CosineAnnealingLR::CosineAnnealingLR() : LearningRateScheduler()
{
	// cycle rate of cosine funtion, usually we use data set of 60000 data sample and 1 sample each update time
	this -> curr_cycle = 0.0f;
    this -> cycle_rate = 1.0f / 60000.0f;
}

CosineAnnealingLR::CosineAnnealingLR(Scalar _cycle_rate) : LearningRateScheduler(), cycle_rate(_cycle_rate)
{
    this -> curr_cycle = 0.0f;
}

CosineAnnealingLR::CosineAnnealingLR(Scalar _learnRate, Scalar _cycle_rate) : LearningRateScheduler(_learnRate), cycle_rate(_cycle_rate)
{
    this -> curr_cycle = 0.0f;
}

CosineAnnealingLR::~CosineAnnealingLR()
{
    // Deconstructor
}

void CosineAnnealingLR::updateLearningRate()
{
	// Update learning rate rule
	this -> curr_cycle += 1.0f;
    this -> lr = this -> lr * exp(-cycle_rate * curr_cycle);
}