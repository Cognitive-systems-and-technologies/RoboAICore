#ifndef NET_H
#define NET_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Interfaces.h"
#include "Dense.h"
#include "Input.h"
#include "MSE.h"
#include "TanhA.h"
#include "Conv2d.h"
#include "MaxPool2d.h"
#include "Relu.h"

typedef struct _Net
{
	Layer** Layers;
	int n_layers;

	Tensor* (*NetForward) (struct _Net* n, Tensor* x);
	void (*NetBackward) (struct _Net* n, Tensor *y);
}Net;

Net Net_Create();
Layer* Net_AddLayer(Net *n, Layer* l);

void Backward_Layer (Layer* l, Tensor *y);
Tensor *Forward_Layer(Layer* l, Tensor* x);

Tensor* Seq_Forward(Net* n, Tensor* x);
void Seq_Backward(Net* n, Tensor* y);

#ifdef __NVCC__
Net Net_CreateGPU();
Tensor* Forward_LayerGPU(Layer* l, Tensor* x);
void Backward_LayerGPU(Layer* l, Tensor* y);
Tensor* Seq_ForwardGPU(Net* n, Tensor* x);
void Seq_BackwardGPU(Net* n, Tensor* y);
#endif // __NVCC__


#ifdef __cplusplus
}
#endif

#endif
