#ifndef TANHA_H
#define TANHA_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"

Layer *TanhA_Create(shape in_shape);
Tensor *TanhA_Forward(Layer* l, Tensor* x, int is_train);
float TanhA_Backward(Layer* l, Tensor* y);

#ifdef __cplusplus
}
#endif

#endif
