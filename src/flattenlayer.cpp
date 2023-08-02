#include <flattenlayer.h>
// flatten config implementation
// flatten config constructor and destructor
FlattenConfig::FlattenConfig()
{
    // Do nothing
}
FlattenConfig::~FlattenConfig()
{
    // Do nothing
}

// flatten config method
uint32_t &FlattenConfig::inputHeightRef()
{
    return this -> inputHeight;
}
uint32_t &FlattenConfig::inputWidthRef()
{
    return this -> inputWidth;
}
uint32_t &FlattenConfig::inputDepthRef()
{
    return this -> inputDepth;
}
// flatten layer implementation
// flatten layer constructor
FlattenLayer::FlattenLayer(FlattenConfig * _config) : Layer(), config(_config)
{
    // create storage for output, the output matrix only 1 layer for dense layer
    output.push_back(new Matrix(config->inputHeight, config->inputWidth * config->inputDepth));
    output.back() -> setZero();
} 

// flatten layer desutructor
FlattenLayer::~FlattenLayer()
{
    // delete output matrix vector
    while(output.size() != 0){
        delete output.back();
        output.pop_back();
    }
}

// flatten layer method for io access
std::vector<Matrix *> &FlattenLayer::inputRef()
{
    return this -> input;
}
std::vector<Matrix *> &FlattenLayer::outputRef()
{
    return this -> output;
}

// flatten layer propagate forward
void FlattenLayer::propagateForward(std::vector<Matrix *> * input)
{
    // loop throught all feature map in input matrix vector
    output.back() -> resize(config->inputHeight, config->inputWidth * config->inputDepth);
    for(size_t i = 0; i < config -> inputDepth; i++)
    {
        output.back() -> block(0, i * config -> inputWidth, config -> inputHeight, config -> inputWidth) = *(*input)[i];
    }

    // flatten input
    output.back() -> resize(1, config -> inputHeight * config -> inputWidth * config -> inputDepth);
}

// flatten layer propagate backward
void FlattenLayer::propagateBackward(std::vector<Matrix *> * errors)
{
    // resize error matrix and output back
    errors -> back() -> resize(config -> inputHeight,  config -> inputWidth * config -> inputDepth);

    std::vector<Matrix *> temp;
    for(size_t i = 0; i < config -> inputDepth; i++)
    {
        // propagate the errors back
        temp.push_back(new Matrix(config -> inputHeight, config -> inputWidth));
        *temp.back() = errors -> back() -> block(0, i * config -> inputWidth, config -> inputHeight, config -> inputWidth);
    }

    // delete previous errors matrix
    while(errors -> size() != 0){
        delete errors -> back();
        errors -> pop_back();
    }
    // return new error matrix
    for(size_t i = 0; i < config -> inputDepth; i++)
    {
        // copy each matrix element pointer in temp;
        errors -> push_back(temp[i]);
    }
}