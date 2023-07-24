#ifndef TENSOR_H
#define TENSOR_H

#ifdef __cplusplus
extern "C" 
{
#endif 
#define DBL_MAX 1.7976931348623158e+308 /* max value */
#define DBL_MIN 2.2250738585072014e-308 /* min positive value */

#define FLT_MAX 3.402823466e+38F /* max value */
#define FLT_MIN 1.175494351e-38F /* min positive value */

#ifdef __NVCC__
#define KERNEL_CALL(x, y) <<<x,y>>>
#define KERNEL_CALL_ONCE <<<1,1>>>
#endif 

#include "TWeightsInit.h"
#include "cJSON.h"

typedef struct shape
{
	int w;//width
	int h;//heigth
	int d;//depth
}shape;

typedef struct shape2
{
	int w;//width
	int h;//heigth
}shape2;

typedef struct Tensor
{
	shape s;
	int n;

	float *w;
	float *dw;
	//additions for optimizer
	//float *vt;
	float sumdw;

	void* tData;//training data
}Tensor;

Tensor Tensor_Create(shape s, float c);
Tensor* Tensor_CreateDyn(shape s, float c);
void Tensor_CopyData(Tensor* dst, Tensor* src);
int tIdx(shape s, int w, int h, int d);
void Tensor_Xavier_Rand(float* w, int n);
void Tensor_He_Rand(float* w, int n);
#ifdef __NVCC__
__global__ void Tensor_FillKernel(int limit, float* w, float v);
void Tensor_FillGPU(Tensor* v, float c);
void Tensor_FillArrayGPU(float* v, int n, float c);
Tensor Tensor_CreateGPU(shape s, float c);
void Tensor_FreeGPU(Tensor* v);
void Tensor_CopyDataGPU(Tensor* dst, Tensor* src);
__global__ void xavier_rand_kernel(void* globalState, float* w, int n);
__global__ void setup_rng_kernel(int limit, void* state);
void Tensor_Xavier_RandGPU(float* w, int n);
__global__ void TPrintKernel(float* w, int n);
void Tensor_PrintGPU(Tensor* v);
void Tensor_PrintArrayGPU(float* v, int n);
#endif 
//=======================================================================
void Tensor_Free(Tensor *v);
float Tensor_Get(Tensor *vol, int x, int y, int d);
void Tensor_Set(Tensor *vol, int w, int h, int d, float v);
void Tensor_Copy(Tensor* dst, Tensor *src);
shape T_Argmax(Tensor *t);

float T_MinValue(Tensor* t);
float T_MaxValue(Tensor* t);
float T_Mean(Tensor* t);

cJSON* Shape_To_JSON(shape s);
cJSON* Tensor_To_JSON(Tensor* v);
void Tensor_Load_JSON(Tensor* t, cJSON* node);
void Tensor_Print(Tensor* x);
#ifdef __cplusplus
}
#endif

#endif
