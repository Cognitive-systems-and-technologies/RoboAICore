#ifndef INPUT_H
#define INPUT_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"

Layer *Input_Create(shape out_shape);
Tensor *Input_Forward(Layer* l, Tensor* x, int is_train);
#ifdef __cplusplus
}
#endif

#endif
