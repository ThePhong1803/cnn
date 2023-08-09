#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include <common.h>
#include <interface.h>

/* Base class prototype declaration */
class Optimizer 
{
    public:
    virtual void DenseOptimizer(DenseLayer * layer);
    virtual void ConvOptimizer(ConvolutionalLayer * layer);
};

/* Stochastic Gradient Descent Implementation */
class SDG : public Optimizer 
{
    void DenseOptimizer(DenseLayer * layer) override;
    void ConvOptimizer(ConvolutionalLayer * layer) override;
};


#endif