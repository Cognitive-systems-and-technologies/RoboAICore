#include "Tensor4.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __NVCC__
Tensor4 Tensor4_CreateGPU(shape4 s, float c) 
{
	Tensor4 v;
	v.s.w = s.w;
	v.s.h = s.h;
	v.s.d = s.d;
	v.s.b = s.b;

	v.n = s.w * s.h * s.d * s.b;
	v.w = NULL; v.dw = NULL; v.vt = NULL;

	if (cudaMalloc((void**)&v.w, v.n * sizeof(float)) != cudaSuccess) printf("Tensor4 weights allocation error\n");
	else Tensor_FillArrayGPU(v.w, v.n, c);
	if (cudaMalloc((void**)&v.dw, v.n * sizeof(float)) != cudaSuccess) printf("Tensor4 grads allocation error\n");
	else cudaMemset(v.dw, 0, sizeof(float) * v.n);
	if (cudaMalloc((void**)&v.vt, v.n * sizeof(float)) != cudaSuccess) printf("Tensor4 additions allocation error\n");
	else cudaMemset(v.vt, 0, sizeof(float) * v.n);

	v.sumdw = 0;
	return v;
}
#endif 
