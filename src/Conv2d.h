#ifndef CONV2D_H
#define CONV2D_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Tensor4.h"
#include "Interfaces.h"

typedef struct Conv2d
{
	Tensor *kernels;
	Tensor biases;
	int n;
	shape2 k_size;
	shape2 stride;
	int pad;
}Conv2d;

Layer* Conv2d_Create(int num_kernels, shape2 k_size, shape2 stride, int pad, RandType weightInit, Layer* in);
Tensor* Conv2d_Forward(Layer* l);
void Conv2d_Backward(Layer* l);
void Conv2d_Free(Layer* l);

cJSON* Conv2d_To_JSON(Conv2d* d);
void Conv2d_Load_JSON(Conv2d* d, cJSON* node);
#ifdef __NVCC__
typedef struct Conv2dGPU
{
	Tensor4 kernels;
	Tensor biases;

	shape2 k_size;
	shape2 stride;
	int pad;
}Conv2dGPU;

Layer* Conv2d_CreateGPU(int num_kernels, shape2 k_size, shape2 stride, int pad, Layer* in);
__global__ void Conv2d_ForwardKernels(shape limit, float* xw, float* kerw, float* bw, float* outw, shape ishape, shape4 kshape, shape oshape, shape2 k_size, shape2 stride, int pad);
Tensor* Conv2d_ForwardGPU(Layer* l);
__global__ void Conv2d_BackwardKernels(shape limit, float* xw, float* xdw, float* kerw, float* kerdw, float* outdw, float* bdw, shape ishape, shape4 kshape, shape oshape, shape2 k_size, shape2 stride, int pad);
void Conv2d_BackwardGPU(Layer* l);
#endif // __NVCC__

#ifdef __cplusplus
}
#endif

#endif
