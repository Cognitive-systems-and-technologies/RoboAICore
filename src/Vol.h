#ifndef VOLUME_H
#define VOLUME_H

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

typedef struct Vol
{
	shape s;
	int n;

	float *w;
	float *dw;
	//additions for optimizer
	float *gsum;
}Vol;

Vol *Vol_Create(shape s, float c, int isTrain);
void Vol_Init(Vol* v, shape s, float c, int isTrain);
void Vol_Free(Vol *v);

cJSON* Vol_To_JSON(Vol *v);

//vol functions
float Vol_WeightedSum(Vol* v1, Vol *v2);

float Vol_Get(Vol *vol, int x, int y, int d);
void Vol_Set(Vol *vol, int w, int h, int d, float v);
void Vol_Copy(Vol* dst, Vol *src);

#ifdef __NVCC__
//fuctions for cuda compiler
__global__ void Vol_PrintKernel(Vol* v);
void Vol_Print(Vol* v);
#endif 

#ifdef __cplusplus
}
#endif

#endif
