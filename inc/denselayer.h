#ifndef __DENSELAYER_H__
#define __DENSELAYER_H__

#include <common.h>
#include <interface.h>
#include <activation.h>
#include <optimizer.h>

// dense layer configuration class
class DenseConfig : public LayerConfig
{
    public:
    // layer parameter
    uint32_t inputWidth;
    uint32_t outputWidth;

    // layer activation function config
    Scalar learningRate;
	ScalarFunPtr actFun;
	ScalarFunPtr dactFun;
    Optimizer * opt;

    DenseConfig();
    ~DenseConfig();

    uint32_t &inputWidthRef() override;
    uint32_t &outputWidthRef() override;

    ScalarFunPtr &activationFunctionRef() override;
	ScalarFunPtr &activationFunctionDerivativeRef() override;
};

// class dense layer prototype
class DenseLayer : public Layer
{
    public:
    DenseConfig *         config;  // layer configuration attribute
    Matrix *              weight;  // weight matrix for this layer
    Matrix *              biases;  // bias matrix for this layer
    Matrix *              dweight; // weight matrix for this layer
    Matrix *              dbiases; // bias matrix for this layer
    std::vector<Matrix *> input;
    std::vector<Matrix *> caches;
    std::vector<Matrix *> output; // output of this layer, used vector but actually only have 1 element for compability with prev layer

    DenseLayer(DenseConfig * _config);
    ~DenseLayer();

    // main method for this layer
    void propagateForward(std::vector<Matrix *> * input) override;
    void propagateBackward(std::vector<Matrix *> * errors) override;
    void updateWeightsAndBiases(int batch_size) override;

    // io access method
    std::vector<Matrix *> &inputRef() override;
    std::vector<Matrix *> &outputRef() override;
};

#endif