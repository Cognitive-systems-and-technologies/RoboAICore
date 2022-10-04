#ifndef SOFTMAX_H
#define SOFTMAX_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Vol.h"
#include "Interfaces.h"

typedef struct Softmax
{
	float* es;
}Softmax;

Layer *Softmax_Create(shape in_shape);
Vol *Softmax_Forward(Layer* l, Vol* x, int is_train);
float Softmax_Backward(Layer* l, Vol* y);

#ifdef __cplusplus
}
#endif

#endif
