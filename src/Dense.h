#ifndef DENSE_H
#define DENSE_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Vol.h"
#include "Interfaces.h"

typedef struct Dense
{
	//shape out_shape;

	//Vol *input;
	//Vol *output;

	float l1_decay_mul;
	float l2_decay_mul;

	//int n_inputs;

	Vol* filters; //List<Vol> filters;
	int n_filters;//store filters lenght

	Vol *biases;
}Dense;

Layer *Dense_Create(int num_neurons, shape in_shape);
Vol *Dense_Forward(Layer* l, Vol* x, int is_train);
float Dense_Backward(Layer* l, Vol* y);

void Dense_Free(Dense *l);
#ifdef __cplusplus
}
#endif

#endif
