#ifndef TENSOR4_H
#define TENSOR4_H

#ifdef __cplusplus
extern "C" 
{
#endif 


#include "TWeightsInit.h"
#include "Tensor.h"
typedef struct shape4
{
	int w;//width
	int h;//heigth
	int d;//depth
	int b;
}shape4;

typedef struct Tensor4
{
	shape4 s;
	int n;

	float *w;
	float *dw;
	//additions for optimizer
	float *vt;
	float sumdw;
}Tensor4;

Tensor4 Tensor4_Create(shape4 s, float c);
void Tensor4_CopyData(Tensor4* dst, Tensor4* src);
int tIdx4(shape4 s, int w, int h, int d, int b);

void Tensor4_Set(Tensor4* t, int w, int h, int d, int b, float v);
float Tensor4_Get(Tensor4* t, int w, int h, int d, int b);
//=======================================================================
void Tensor4_Free(Tensor4 *v);
void Tensor4_Copy(Tensor4* dst, Tensor4 *src);
void T4Print(Tensor4* t);

#ifdef __NVCC__
Tensor4 Tensor4_CreateGPU(shape4 s, float c);
#endif // __NVCC__


#ifdef __cplusplus
}
#endif

#endif
