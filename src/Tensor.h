#ifndef TENSOR_H
#define TENSOR_H

#ifdef __cplusplus
extern "C" 
{
#endif 

#include "cJSON.h"

#define DBL_MAX 1.7976931348623158e+308 /* max value */
#define DBL_MIN 2.2250738585072014e-308 /* min positive value */

#define FLT_MAX 3.402823466e+38F /* max value */
#define FLT_MIN 1.175494351e-38F /* min positive value */

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
Tensor* Tensor_CreateCopy(Tensor* v);
void Tensor_Init(Tensor* v, shape s, float c, int isTrain);
void Tensor_Free(Tensor *v);

cJSON* Tensor_To_JSON(Tensor *v);

//vol functions
float Tensor_WeightedSum(Tensor* v1, Tensor *v2);

float Tensor_Get(Tensor *vol, int x, int y, int d);
void Tensor_Set(Tensor *vol, int w, int h, int d, float v);
void Tensor_Copy(Tensor* dst, Tensor *src);
shape T_Argmax(Tensor *t);

#ifdef __NVCC__
//fuctions for cuda compiler
__global__ void Tensor_PrintKernel(Tensor* v);
void Tensor_Print(Tensor* v);
#endif 

#ifdef __cplusplus
}
#endif

#endif
