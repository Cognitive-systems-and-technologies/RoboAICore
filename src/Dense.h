#ifndef DENSE_H
#define DENSE_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"
#include "dList.h"

typedef struct Dense
{
	//LayerActivation activation;
	Tensor *kernels;
	Tensor biases;
	int n;
}Dense;

//Layer *Dense_Create(int num_neurons, RandType weightInit, LayerActivation act, Layer *in);
Layer *Dense_Create(int num_neurons, RandType weightInit, Layer *in);
Tensor* Dense_Forward(Layer* l);
void Dense_Backward(Layer* l);

void Dense_Free(Layer* l);

cJSON* Dense_To_JSON(Dense* d);
void Dense_Load_JSON(Dense* d, cJSON* node);
void Dense_GetGrads(Dense* l, dList* grads);
#ifdef __NVCC__
Layer* Dense_CreateGPU(int num_neurons, Layer* in);
__global__ void Dense_ForwardKernels(shape limit, float* x, float* k, float* out, shape s);
Tensor* Dense_ForwardGPU(Layer* l);
__global__ void Dense_BackwardKernels(shape limit, float* xw, float* xdw, float* kw, float* kdw, float* bdw, float* outdw, shape s);
void Dense_BackwardGPU(Layer* l);
#endif // __NVCC__

#ifdef __cplusplus
}
#endif

#endif
