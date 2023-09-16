#include "Utils.h"

#ifdef __NVCC__
float* createFloatArrayGPU(int n)
{
	float* a = NULL;
	if (cudaMalloc((void**)&a, n * sizeof(float)) != cudaSuccess) {
		printf("Array GPU allocation error\n");
		return NULL;
	}
	else {
		cudaMemset(a, 0, sizeof(float) * n);
		return a;
	}
}
#endif // __NVCC__