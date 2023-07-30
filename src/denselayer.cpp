#include <denselayer.h>

// class dense layer config
DenseConfig::DenseConfig()
{
    // Do nothing
}

DenseConfig::~DenseConfig()
{
    // Do nothing
}

// DenseConfig method for base class
uint32_t &DenseConfig::inputWidthRef()
{
    return this -> inputWidth;
}

uint32_t &DenseConfig::outputWidthRef()
{
    return this -> outputWidth;
}

ScalarFunPtr &DenseConfig::activationFunctionRef()
{
    return this -> actFun;
}
ScalarFunPtr &DenseConfig::activationFunctionDerivativeRef()
{
    return this -> dactFun;
}
// class dense layer implementation
// class dense layer contructor and destructor
DenseLayer::DenseLayer(DenseConfig * _config) : config(_config)
{
    this -> weight = new Matrix(config -> inputWidth, config -> outputWidth);
    this -> biases = new Matrix(1, config -> outputWidth);
    this -> output.push_back(new Matrix(1, config -> outputWidth));
    this -> caches.push_back(new Matrix(1, config -> outputWidth));

    // init weight and biases
    this -> biases -> setZero();
    this -> weight -> setRandom();
    this -> output.back() -> setZero();
    this -> caches.back() -> setZero();
}

DenseLayer::~DenseLayer()
{
    delete this -> weight;
    delete this -> biases;
    while(output.size() != 0)
    {
        delete output.back();
        output.pop_back();
    }
    while(caches.size() != 0)
    {
        delete caches.back();
        caches.pop_back();
    }
}

// io access method
std::vector<Matrix *> &DenseLayer::inputRef()
{
    return this -> input;
}

std::vector<Matrix *> &DenseLayer::outputRef()
{
    return this -> output;
}

void DenseLayer::propagateForward(std::vector<Matrix *> * input)
{
    // check input vector size
    assert(input -> size() == 1 && output.size() == 1 && caches.size() == 1); // only 1 matrix element is allowed
    // calculate weighted output;
    (*caches.back()) = (*input -> back()) * (*weight);
    // apply activation fucntion
    (*output.back()) = caches.back() -> unaryExpr(std::ptr_fun(config -> actFun));
}

void DenseLayer::propagateBackward(std::vector<Matrix *> * errors)
{
    // check erros vector size
    assert(errors -> size() == 1 && output.size() == 1 && caches.size() == 1);
    // calcualte error signal
    Matrix delta = (errors -> back() -> array()) * (caches.back() -> array().unaryExpr(std::ptr_fun(config -> dactFun)));
    // prepare error for prev layer
    (*errors -> back()) = (*errors -> back()) * (weight -> transpose());

    // update weight and bias, we will optimize this with mini-batches
    (*weight) += config -> learningRate * (input.back() -> transpose() * delta);
    (*biases) += config -> learningRate * delta;
}