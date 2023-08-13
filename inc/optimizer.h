#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include <common.h>
#include <interface.h>
#include <denselayer.h>
#include <convolutionallayer.h>

/* Stochastic Gradient Descent Implementation */
class SGD : public Optimizer 
{
	Scalar learnRate;
    Scalar momentum;
    LearningRateScheduler * lr_scheduler;
    public:
	SGD();
	SGD(Scalar learnRate, Scalar _momentum);
    SGD(Scalar learnRate, Scalar _momentum, LearningRateScheduler * _lr_scheduler);
    ~SGD();
    void DenseOptimizer(DenseLayer * layer, int batch_size) override;
    void ConvOptimizer(ConvolutionalLayer * layer, int batch_size) override;
    void ScheduleLearningRate(Scalar step) override;
    Scalar getLearningRate() override;
};

class ExponentDecayLearnRate : public LearningRateScheduler
{
    public:
    Scalar decay_factor;
    ExponentDecayLearnRate();
    ExponentDecayLearnRate(Scalar _decay_factor);
    ExponentDecayLearnRate(Scalar _learnRate, Scalar _decay_factor);
    ~ExponentDecayLearnRate();

    // override base method
    void updateLearningRate(Scalar step) override;
};


#endif