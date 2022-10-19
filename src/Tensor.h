#ifndef TENSOR_H
#define TENSOR_H

#ifdef __cplusplus
extern "C" 
{
#endif 

#include "cJSON.h"

#ifdef __NVCC__
#define KERNEL_CALL(x, y) <<<x,y>>>
#define KERNEL_CALL_ONCE <<<1,1>>>
#endif 

typedef struct shape
{
	int w;//width
	int h;//heigth
	int d;//depth
}shape;

typedef struct Tensor
{
	shape s;
	int n;

	float *w;
	float *dw;
	//additions for optimizer
	float *gsum;
}Tensor;

Tensor *Tensor_Create(shape s, float c, int isTrain);
void Tensor_Init(Tensor* v, shape s, float c, int isTrain);
void Tensor_Free(Tensor *v);

cJSON* Tensor_To_JSON(Tensor *v);

//vol functions
float Tensor_WeightedSum(Tensor* v1, Tensor *v2);

float Tensor_Get(Tensor *vol, int x, int y, int d);
void Tensor_Set(Tensor *vol, int w, int h, int d, float v);
void Tensor_Copy(Tensor* dst, Tensor *src);

#ifdef __NVCC__
//fuctions for cuda compiler
__global__ void Tensor_PrintKernel(Tensor* v);
void Tensor_Print(Tensor* v);
#endif 

#ifdef __cplusplus
}
#endif

#endif
