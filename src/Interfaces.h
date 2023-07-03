#ifndef INTERFACES_H
#define INTERFACES_H

#ifdef __cplusplus
extern "C"
{
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "Tensor.h"

typedef enum LayerType {
	LT_INPUT,
	LT_DENSE,
	LT_RELU,
	LT_SOFTMAX,
	LT_REGRESSION,
	LT_SVM,
	LT_CONV,
	LT_MAXPOOL,
	LT_MSE,
	LT_TANHA
} LayerType;

typedef struct Layer
{
	shape out_shape;
	int n_inputs;
	LayerType type;

	Tensor* input;
	Tensor output;

	void* aData;//additional layer data
}Layer;

typedef struct LData
{
	float loss;
}LData;

#ifdef __cplusplus
}
#endif

#endif
