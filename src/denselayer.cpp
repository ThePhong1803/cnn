#include <denselayer.h>

// class dense layer config
DenseConfig::DenseConfig()
{
    // Do nothing
}

DenseConfig::~DenseConfig()
{
    // Delete optimizer object
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
    this -> dweight = new Matrix(config -> inputWidth, config -> outputWidth);
    this -> dbiases = new Matrix(1, config -> outputWidth);
	this -> vweight = new Matrix(config -> inputWidth, config -> outputWidth);
    this -> vbiases = new Matrix(1, config -> outputWidth);
    this -> output.push_back(new Matrix(1, config -> outputWidth));
    this -> caches.push_back(new Matrix(1, config -> outputWidth));

    // init weight and biases
    this -> biases -> setZero();
    this -> weight -> setRandom();
    this -> dbiases -> setZero();
    this -> dweight -> setZero();
	this -> vbiases -> setZero();
    this -> vweight -> setZero();
    this -> output.back() -> setZero();
    this -> caches.back() -> setZero();
}

DenseLayer::~DenseLayer()
{
    delete this -> weight;
    delete this -> biases;
    delete this -> dweight;
    delete this -> dbiases;
	delete this -> vweight;
    delete this -> vbiases;
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
    (*caches.back()) = (*input -> back()) * (*weight) + (*biases);
    // apply activation fucntion
    (*output.back()) = caches.back() -> unaryExpr([this](Scalar x) {return this -> config -> actFun(x);});
}

void DenseLayer::propagateBackward(std::vector<Matrix *> * errors)
{
    // check erros vector size
    assert(errors -> size() == 1 && output.size() == 1 && caches.size() == 1);
    // calcualte error signal
    Matrix delta = Matrix((errors -> back() -> array()) * (caches.back() -> unaryExpr([this](Scalar x) {return this -> config -> dactFun(x);}).array()));
    // prepare error for prev layer
    (*errors -> back()) = delta * (weight -> transpose());

    // update weight and bias, we will optimize this with mini-batches
    (*dweight) += (input.back() -> transpose() * delta);
    (*dbiases) += delta;
}

void DenseLayer::updateWeightsAndBiases(int batch_size, Optimizer * optimizer) 
{
    /* Usually call function from optimier */
    // for testing we gonnal do it here
    optimizer-> DenseOptimizer(this, batch_size);
}