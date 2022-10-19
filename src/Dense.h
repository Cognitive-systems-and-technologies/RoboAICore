#ifndef DENSE_H
#define DENSE_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"

typedef struct Dense
{
	//shape out_shape;

	//Tensor *input;
	//Tensor *output;

	float l1_decay_mul;
	float l2_decay_mul;

	//int n_inputs;

	Tensor* filters; //List<Tensor> filters;
	int n_filters;//store filters lenght

	Tensor *biases;
}Dense;

Layer *Dense_Create(int num_neurons, shape in_shape);
Tensor *Dense_Forward(Layer* l, Tensor* x, int is_train);
float Dense_Backward(Layer* l, Tensor* y);

void Dense_Free(Dense *l);
#ifdef __cplusplus
}
#endif

#endif
