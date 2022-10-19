#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Net.h"

Tensor* Seq_Forward(Net* n, Tensor* x, int is_training);
float Seq_Backward(Net* n, Tensor* y);
#ifdef __cplusplus
}
#endif

#endif
