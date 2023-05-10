#ifndef DENSE_H
#define DENSE_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"
#include "TCommon.h"
#include "TWeightsInit.h"
#include "dList.h"

typedef struct Dense
{
	//shape out_shape;

	//Tensor *input;
	//Tensor *output;

	float l1_decay_mul;
	float l2_decay_mul;

	//int n_inputs;

	Tensor* kernels; //List<Tensor> kernels;
	int n_kernels;//store kernels lenght

	Tensor *biases;
}Dense;

Layer *Dense_Create(int num_neurons, shape in_shape);
Tensor *Dense_Forward(Layer* l, Tensor* x, int is_train);
float Dense_Backward(Layer* l, Tensor* y);

void Dense_GetGrads(Dense* l, dList *grads);
void Dense_Free(Dense *l);

cJSON* Dense_To_JSON(Dense* d);
void Dense_Load_JSON(Dense* d, cJSON *node);
#ifdef __cplusplus
}
#endif

#endif
