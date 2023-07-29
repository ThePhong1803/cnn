#ifndef __FLATTENLAYER_H__
#define __FLATTENLAYER_H__

#include <common.h>
#include <interface.h>

// class layer config
class FlattenConfig : public LayerConfig
{
	public:
	// image parameter
	uint32_t inputHeight;
	uint32_t inputWidth;
	uint32_t inputDepth;	
	public:
	FlattenConfig();
	~FlattenConfig();
	uint32_t &inputHeightRef() override;
	uint32_t &inputWidthRef() override;
	uint32_t &inputDepthRef() override;
};

// class protorype
class FlattenLayer : public Layer
{
    public:
    FlattenConfig *       config;
    std::vector<Matrix *> input;
    std::vector<Matrix *> output; // store input and tranform it to flattened output

    // layer constuctor and destructor
    FlattenLayer(FlattenConfig * _config);
    ~FlattenLayer();

    // implement layer method
    void propagateForward(std::vector<Matrix *> * input) override;
    void propagateBackward(std::vector<Matrix *> * errors) override;

    // implement layer
    std::vector<Matrix *> &inputRef();
    std::vector<Matrix *> &outputRef(); 
};

#endif