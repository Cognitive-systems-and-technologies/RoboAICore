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

Layer *MSE_Create(Layer* in);
Tensor *MSE_Forward(Layer* l);
void MSE_Backward(Layer* l, Tensor* y_true);

#ifdef __NVCC__
Layer* MSE_CreateGPU(Layer* in);
Tensor* MSE_ForwardGPU(Layer* l);
__global__ void MSE_BackwardKernels(int limit, float* xw, float* xdw, float* yw, float n, float* sum);
void MSE_BackwardGPU(Layer* l, Tensor* y_true);
#endif // __NVCC__

#ifdef __cplusplus
}
#endif

#endif
