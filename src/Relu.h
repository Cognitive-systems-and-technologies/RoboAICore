#ifndef RELU_H
#define RELU_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Vol.h"
#include "Interfaces.h"

Layer *Relu_Create(shape out_shape);
Vol *Relu_Forward(Layer* l, Vol* x, int is_train);
float Relu_Backward(Layer* l, Vol* y);
#ifdef __cplusplus
}
#endif

#endif
