#ifndef LOSSES_H
#define LOSSES_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"

#include <stdlib.h>
#include <stdio.h>

	float MSE_Loss(Tensor *y, Tensor *y_true);
	Tensor SoftmaxProb(Tensor* t);
	float Cross_entropy_Loss(Tensor* y, int idx);
	float Regression_Loss(Tensor* y, int idx, float val);

#ifdef __NVCC__
	float Cross_entropy_LossGPU(Tensor* y, int idx);
	__global__ void Cross_entropy_LossKernels(int n, float* xw, float* ydw, int idx);
	Tensor SoftmaxProbGPU(Tensor* t);
	__global__ void SoftmaxProbKernels(int n, float* iw, float* ow);
	float MSE_LossGPU(Tensor* y, Tensor* y_true);
	__global__ void MSE_LossKernels(int n, float* yw, float* ytw, float* ydw, float* sum);
#endif // __NVCC__

#ifdef __cplusplus
}
#endif

#endif
