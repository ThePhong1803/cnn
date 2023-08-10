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
    public:
	SGD();
	SGD(Scalar learnRate, Scalar _momentum);
    ~SGD();
    void DenseOptimizer(DenseLayer * layer, int batch_size) override;
    void ConvOptimizer(ConvolutionalLayer * layer, int batch_size) override;
};


#endif