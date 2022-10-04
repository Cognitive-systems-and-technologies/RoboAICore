#ifndef INPUT_H
#define INPUT_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Vol.h"
#include "Interfaces.h"

Layer *Input_Create(shape out_shape);
Vol *Input_Forward(Layer* l, Vol* x, int is_train);
#ifdef __cplusplus
}
#endif

#endif
