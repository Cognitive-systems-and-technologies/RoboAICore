#ifndef TANHA_H
#define TANHA_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Interfaces.h"
#include <stdlib.h>
#include <stdio.h>

Layer *TanhA_Create(Layer *in);
Tensor* TanhA_Forward(Layer* l);
void TanhA_Backward(Layer* l);

#ifdef __NVCC__
Layer* TanhA_CreateGPU(Layer* in);
__global__ void TanhA_ForwardKernels(float* xw, float* outw);
Tensor* TanhA_ForwardGPU(Layer* l);
__global__ void TanhA_BackwardKernels(float* xdw, float* outw, float* outdw);
void TanhA_BackwardGPU(Layer* l);
#endif // __NVCC__

#ifdef __cplusplus
}
#endif

#endif
