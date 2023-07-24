#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"

typedef struct MaxPool2d
{
	shape2 k_size;
	shape2 stride;
	int pad;//0,1 or int > 0
}MaxPool2d;

Layer* MaxPool2d_Create(shape2 k_size, shape2 stride, int pad, Layer* in);
Tensor* MaxPool2d_Forward(Layer* l);
void MaxPool2d_Backward(Layer* l);
void MaxPool2d_Free(Layer* l);

#ifdef __NVCC__
Layer* MaxPool2d_CreateGPU(shape2 k_size, shape2 stride, int pad, Layer* in);
__global__ void MaxPool2d_ForwardKernels(shape limit, float* xw, float* outw, shape ishape, shape oshape, shape2 k_size, shape2 stride, int pad);
Tensor* MaxPool2d_ForwardGPU(Layer* l);
__global__ void MaxPool2d_BackwardKernels(shape limit, float* xw, float* xdw, float* outdw, shape ishape, shape oshape, shape2 k_size, shape2 stride, int pad);
void MaxPool2d_BackwardGPU(Layer* l);
#endif // __NVCC__


#ifdef __cplusplus
}
#endif

#endif
