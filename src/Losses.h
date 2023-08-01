#ifndef LOSSES_H
#define LOSSES_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"

#include <stdlib.h>
#include <stdio.h>

	float MSE_Loss(Tensor *y, Tensor *y_true);
	Tensor SoftmaxProp(Tensor* t);
	float Cross_entropy_Loss(Tensor* y, int idx);
	float Regression_Loss(Tensor* y, int idx, float val);
#ifdef __NVCC__
#endif // __NVCC__

#ifdef __cplusplus
}
#endif

#endif
