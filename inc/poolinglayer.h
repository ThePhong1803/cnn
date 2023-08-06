#ifndef __POOLLAYER_H__
#define __POOLLAYER_H__
#include <common.h>
#include <interface.h>
#include <activation.h>

class PoolingConfig : public LayerConfig {
	public:
	// image parameter
	uint32_t inputHeight;
	uint32_t inputWidth;
	uint32_t inputDepth;
	
	// kernel parameter
	uint32_t kernelHeight;
	uint32_t kernelWidth;
	
	public:
	PoolingConfig();
	~PoolingConfig();
	uint32_t &inputHeightRef() override;
	uint32_t &inputWidthRef() override;
	uint32_t &inputDepthRef() override;
	
	uint32_t &kernelHeightRef() override;
	uint32_t &kernelWidthRef() override;
};

// utilities class
template <typename T>
class ElementPooling
{
	public:
	int row;
	int col;
	T value;
	// list of constructor
	ElementPooling(): row(0), col(0), value(0) {}
	ElementPooling(int _row, int _col) : row(_row), col(_col), value(0) {}
	ElementPooling(int _row, int _col, T _value) : row(_row), col(_col), value(_value) {}

	// copy constructor
	ElementPooling(const ElementPooling &obj)
	{
		this -> row = obj.row;
		this -> col = obj.col;
		this -> value = obj.value;
	}
};


class MaxPoolingLayer : public Layer {
	public:
	PoolingConfig *		  		config;
	std::vector<Matrix *> 		input;
	std::vector<Matrix *> 		output;
	std::vector<Matrix *>		kernel;
	// layer constructor and destructor
	MaxPoolingLayer(PoolingConfig * _config);
	~MaxPoolingLayer();

	// layer method
	void propagateForward(std::vector<Matrix*> * input);
	void propagateBackward(std::vector<Matrix*> * errors);

	// layer method for base class pointer
	std::vector<Matrix *> &inputRef() override;
	std::vector<Matrix *> &outputRef() override;
};

class AveragePoolingLayer : public Layer {};

class GlobalPoolingLayer : public Layer {};

#endif