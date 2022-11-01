#ifndef REGRESSION_H
#define REGRESSION_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"

Layer *Regression_Create(shape in_shape);
Tensor *Regression_Forward(Layer* l, Tensor* x, int is_train);
float Regression_Backward(Layer* l, Tensor* y);

#ifdef __cplusplus
}
#endif

#endif
