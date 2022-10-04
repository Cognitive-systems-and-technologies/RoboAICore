#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Net.h"

Vol* Seq_Forward(Net* n, Vol* x, int is_training);
float Seq_Backward(Net* n, Vol* y);
#ifdef __cplusplus
}
#endif

#endif
