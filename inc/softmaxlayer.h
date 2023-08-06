#ifndef __SOFTMAXLAYER_H__
#define __SOFTMAXLAYER_H__

#include <common.h>
#include <interface.h>
#include <activation.h>

// Softmax layer configuration class
class SoftmaxConfig : public LayerConfig
{
    public:
    // layer parameter
    uint32_t inputWidth;
    uint32_t outputWidth;

    // layer activation function config
    Scalar learningRate;

    SoftmaxConfig();
    ~SoftmaxConfig();

    uint32_t &inputWidthRef() override;
    uint32_t &outputWidthRef() override;
};

// class Softmax layer prototype
class SoftmaxLayer : public Layer
{
    public:
    SoftmaxConfig *       config;  // layer configuration attribute
    Matrix *              Identity;
    std::vector<Matrix *> input;
    std::vector<Matrix *> output; // output of this layer, used vector but actually only have 1 element for compability with prev layer

    SoftmaxLayer(SoftmaxConfig * _config);
    ~SoftmaxLayer();

    // main method for this layer
    void propagateForward(std::vector<Matrix *> * input) override;
    void propagateBackward(std::vector<Matrix *> * errors) override;

    // io access method
    std::vector<Matrix *> &inputRef() override;
    std::vector<Matrix *> &outputRef() override;
};

#endif