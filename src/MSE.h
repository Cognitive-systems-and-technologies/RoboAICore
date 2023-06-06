#ifndef MSE_H
#define MSE_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Interfaces.h"
#include "Tensor.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Layer * MSE_Create(shape in_shape);
Tensor * MSE_Forward(Layer* l, Tensor* x, int is_train);
float MSE_Backward(Layer* l, Tensor* y_true);

#ifdef __cplusplus
}
#endif

#endif
