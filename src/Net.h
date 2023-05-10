#ifndef NET_H
#define NET_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"
#include "Dense.h"
#include "Softmax.h"
#include "Input.h"
#include "Relu.h"
#include "Regression.h"
#include "dList.h"

typedef struct _Net
{
	Layer** Layers;
	int n_layers;

	Tensor* (*NetForward) (struct _Net* n, Tensor* x, int is_training);
	float (*NetBackward) (struct _Net* n, Tensor *y);
	void (*NetInit) (shape in);
}Net;

void Net_Init(Net* net, Tensor*(*forward)(Net* n, Tensor* x, int is_training), float (*backward) (Net *n, Tensor *y), void (*init) (shape in));
float Backward_Layer (Layer* l, Tensor *y);
Tensor *Forward_Layer(Layer* l, Tensor* x);
dList Net_getGradients(Net* net);

cJSON* Layer_To_JSON(Layer* l);
void Layer_Load_JSON(Layer* t, cJSON* node);
cJSON* Net_To_JSON(Net* n);
void Net_Load_JSON(Net* t, cJSON* node);
#ifdef __cplusplus
}
#endif

#endif
