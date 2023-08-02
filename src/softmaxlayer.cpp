#include <softmaxlayer.h>

// class Softmax layer config
SoftmaxConfig::SoftmaxConfig()
{
    // Do nothing
}

SoftmaxConfig::~SoftmaxConfig()
{
    // Do nothing
}

// SoftmaxConfig method for base class
uint32_t &SoftmaxConfig::inputWidthRef()
{
    return this -> inputWidth;
}

uint32_t &SoftmaxConfig::outputWidthRef()
{
    return this -> outputWidth;
}

// class Softmax layer implementation
// class Softmax layer contructor and destructor
SoftmaxLayer::SoftmaxLayer(SoftmaxConfig * _config) : config(_config)
{
    this -> output.push_back(new Matrix(1, config -> outputWidth));
    this -> Identity = new Matrix(config -> outputWidth, config -> outputWidth);

    // init weight and biases
    this -> output.back() -> setZero();
    this -> Identity -> setZero();
    // init indentity matrix
    for(size_t i = 0; i < config -> outputWidth; i++) this -> Identity -> coeffRef(i, i) = 1;
}

SoftmaxLayer::~SoftmaxLayer()
{
    while(output.size() != 0)
    {
        delete output.back();
        output.pop_back();
    }
}

// io access method
std::vector<Matrix *> &SoftmaxLayer::inputRef()
{
    return this -> input;
}

std::vector<Matrix *> &SoftmaxLayer::outputRef()
{
    return this -> output;
}

void SoftmaxLayer::propagateForward(std::vector<Matrix *> * input)
{
    // check input vector size
    assert(input -> size() == 1 && output.size() == 1); // only 1 matrix element is allowed
    // calculate  output;
    (*output.back()) = input -> back() -> unaryExpr([](Scalar x) -> Scalar {return exp(x);});
    Scalar sumExp = output.back() -> sum();
    (*output.back()) = output.back() -> unaryExpr([sumExp](Scalar x) -> Scalar {return x / sumExp;});
}

void SoftmaxLayer::propagateBackward(std::vector<Matrix *> * errors)
{
    // check erros vector size
    assert(errors -> size() == 1 && output.size() == 1);
    // calcualte error signal
    Matrix delta(1, config -> outputWidth);
    delta = (*errors -> back());                             
    
    // prepare error for prev layer
    // clear error matrix:
    while(errors -> size() != 0) {
        delete errors -> back();
        errors -> pop_back();
    }

    // create conainter for prev layer matrix
    errors -> push_back(new Matrix(1, config -> outputWidth));

    // create M matrix which have outputwidh duplicate of output vector
    Matrix temp(config -> outputWidth, config -> outputWidth);
    temp = (output.back() -> transpose().rowwise()).replicate(config -> outputWidth);
    std::cout << temp << std::endl;
    temp = temp.array() * ((*this -> Identity) - temp.transpose()).array();
    temp = temp.dot(delta.transpose());
    (*errors -> back()) = temp;
}