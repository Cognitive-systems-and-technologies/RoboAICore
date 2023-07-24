#ifndef SOFTMAX_H
#define SOFTMAX_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"

typedef struct Softmax
{
	float* sums;
}Softmax;

Layer *Softmax_Create(Layer *in);
Tensor *Softmax_Forward(Layer* l);
void Softmax_Backward(Layer* l, Tensor* y);

#ifdef __cplusplus
}
#endif

#endif
