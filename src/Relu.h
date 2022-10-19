#ifndef RELU_H
#define RELU_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"

Layer *Relu_Create(shape out_shape);
Tensor *Relu_Forward(Layer* l, Tensor* x, int is_train);
float Relu_Backward(Layer* l, Tensor* y);
#ifdef __cplusplus
}
#endif

#endif
