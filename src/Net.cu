#include "Net.h"
#include <stdlib.h>

#ifdef __NVCC__
Net Net_CreateGPU()
{
	Net n;
	n.Layers = NULL;
	n.n_layers = 0;
	n.NetForward = Seq_ForwardGPU;
	n.NetBackward = Seq_BackwardGPU;
	return n;
}
Tensor* Forward_LayerGPU(Layer* l, Tensor* x)
{
	Tensor* y = NULL;
	switch (l->type)
	{
	case LT_INPUT: y = Input_ForwardGPU(l, x); break;
	case LT_DENSE: y = Dense_ForwardGPU(l); break;
	case LT_SOFTMAX: break;
	case LT_RELU: y = Relu_ForwardGPU(l); break;
	case LT_REGRESSION: break;
	case LT_MSE: y = MSE_ForwardGPU(l); break;
	case LT_TANHA: y = TanhA_ForwardGPU(l); break;
	case LT_CONV: y = Conv2d_ForwardGPU(l); break;
	case LT_MAXPOOL: y = MaxPool2d_ForwardGPU(l); break;
	default: break;
	}
	return y;
}
void Backward_LayerGPU(Layer* l, Tensor* y)
{
	switch (l->type)
	{
	case LT_DENSE: Dense_BackwardGPU(l); break;
	case LT_SOFTMAX: break;
	case LT_RELU: Relu_BackwardGPU(l); break;
	case LT_REGRESSION: break;
	case LT_MSE: MSE_BackwardGPU(l, y); break;
	case LT_TANHA: TanhA_BackwardGPU(l); break;
	case LT_CONV: Conv2d_BackwardGPU(l); break;
	case LT_MAXPOOL: MaxPool2d_BackwardGPU(l); break;
	default: break;
	}
}
Tensor* Seq_ForwardGPU(Net* n, Tensor* x)
{
	Tensor* y = Forward_LayerGPU(n->Layers[0], x);
	for (int i = 1; i < n->n_layers; i++)
	{
		y = Forward_LayerGPU(n->Layers[i], y);
	}
	return y;
}
void Seq_BackwardGPU(Net* n, Tensor* y)
{
	int N = n->n_layers;
	for (int i = N - 1; i >= 0; i--)
	{
		Backward_LayerGPU(n->Layers[i], y);
	}
}
#endif // __NVCC__