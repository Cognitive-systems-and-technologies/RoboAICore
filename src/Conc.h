#ifndef CONCATE_H
#define CONCATE_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"
#include "dList.h"

//Layer *Dense_Create(int num_neurons, RandType weightInit, LayerActivation act, Layer *in);
Layer* Conc_Create(Layer* in1, Layer* in2);
Tensor* Conc_Forward(Layer* l);
void Conc_Backward(Layer* l);
void Conc_BackpropGrads(Layer* l, Tensor* t1, Tensor* t2);
#ifdef __cplusplus
}
#endif

#endif
