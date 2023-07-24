#ifndef REGRESSION_H
#define REGRESSION_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Interfaces.h"
#include "Tensor.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Layer *Regression_Create(Layer *in);
Tensor *Regression_Forward(Layer* l);
void Regression_Backward(Layer* l, Tensor* y);

#ifdef __cplusplus
}
#endif

#endif
