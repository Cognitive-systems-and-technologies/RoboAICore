#ifndef NET_H
#define NET_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Vol.h"
#include "Interfaces.h"
#include "Dense.h"
#include "Softmax.h"
#include "Input.h"
#include "Relu.h"

typedef struct _Net
{
	Layer** Layers;
	int n_layers;

	Vol* (*NetForward) (struct _Net* n, Vol* x, int is_training);
	float (*NetBackward) (struct _Net* n, Vol *y);
	void (*NetInit) (shape in);
}Net;

void Net_Init(Net* net, Vol*(*forward)(Net* n, Vol* x, int is_training), float (*backward) (Net *n, Vol *y), void (*init) (shape in));
float Backward_Layer (Layer* l, Vol *y);
Vol *Forward_Layer(Layer* l, Vol* y);
#ifdef __cplusplus
}
#endif

#endif
