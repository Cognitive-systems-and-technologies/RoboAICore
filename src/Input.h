#ifndef INPUT_H
#define INPUT_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"

#include <stdlib.h>
#include <stdio.h>

Layer *Input_Create(shape out_shape);
Tensor *Input_Forward(Layer* l);

void Input_Free(Layer *l);
#ifdef __NVCC__
Layer* Input_CreateGPU(shape out_shape);
Tensor* Input_ForwardGPU(Layer* l, Tensor* x);
#endif // __NVCC__

#ifdef __cplusplus
}
#endif

#endif
