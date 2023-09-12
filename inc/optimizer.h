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
    void ScheduleLearningRate() override;
    Scalar getLearningRate() override;
	void setLearningRate(Scalar _new_lr) override;
};

class ExponentDecayLearnRate : public LearningRateScheduler
{
    public:
	Scalar lr_cnt;
    Scalar decay_factor;
    ExponentDecayLearnRate();
    ExponentDecayLearnRate(Scalar _decay_factor);
    ExponentDecayLearnRate(Scalar _learnRate, Scalar _decay_factor);
    ~ExponentDecayLearnRate();

    // override base method
    void updateLearningRate() override;
	void resetStep() override;
};

class CosineAnnealingLR : public LearningRateScheduler
{
    public:
	Scalar curr_cycle;
    Scalar cycle_rate;
    CosineAnnealingLR();
    CosineAnnealingLR(Scalar _cycle_rate);
    CosineAnnealingLR(Scalar _learnRate, Scalar _cycle);
    ~CosineAnnealingLR();

    // override base method
    void updateLearningRate() override;
};
#endif