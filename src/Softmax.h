#ifndef SOFTMAX_H
#define SOFTMAX_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"

typedef struct Softmax
{
	float* es;
}Softmax;

Layer *Softmax_Create(shape in_shape);
Tensor *Softmax_Forward(Layer* l, Tensor* x, int is_train);
float Softmax_Backward(Layer* l, Tensor* y);

#ifdef __cplusplus
}
#endif

#endif
