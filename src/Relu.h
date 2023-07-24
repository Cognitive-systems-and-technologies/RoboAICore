#ifndef RELU_H
#define RELU_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"

Layer *Relu_Create(Layer *in);
Tensor *Relu_Forward(Layer* l);
void Relu_Backward(Layer* l);

#ifdef __NVCC__
Layer* Relu_CreateGPU(Layer* in);
__global__ void Relu_ForwardKernels(int limit, float* xw, float* outw);
Tensor* Relu_ForwardGPU(Layer* l);
__global__ void Relu_BackwardKernels(int limit, float* xdw, float* outw, float* outdw);
void Relu_BackwardGPU(Layer* l);
#endif // __NVCC__

#ifdef __cplusplus
}
#endif

#endif
