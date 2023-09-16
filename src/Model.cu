#include "Model.h"
#include <stdlib.h>

#ifdef __NVCC__
Model Model_CreateGPU()
{
	Model n;
	n.Layers = NULL;
	n.n_layers = 0;
	n.NetForward = NULL;//Seq_ForwardGPU;
	n.NetBackward = NULL;// Seq_BackwardGPU;
	return n;
}
Tensor* Forward_LayerGPU(Layer* l)
{
	Tensor* y = NULL;
	switch (l->type)
	{
	case LT_INPUT: y = Input_ForwardGPU(l); break;
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
void Backward_LayerGPU(Layer* l)
{
	switch (l->type)
	{
	case LT_DENSE: Dense_BackwardGPU(l); break;
	case LT_SOFTMAX: break;
	case LT_RELU: Relu_BackwardGPU(l); break;
	case LT_REGRESSION: break;
	case LT_MSE: break;//MSE_BackwardGPU(l, y); break;
	case LT_TANHA: TanhA_BackwardGPU(l); break;
	case LT_CONV: Conv2d_BackwardGPU(l); break;
	case LT_MAXPOOL: MaxPool2d_BackwardGPU(l); break;
	default: break;
	}
}

void Model_ForwardGPU(Model* n)
{
	for (int i = 0; i < n->n_layers; i++)
	{
		Forward_LayerGPU(n->Layers[i]);
	}
}

void Model_BackwardGPU(Model* n)
{
	int N = n->n_layers;
	for (int i = N - 1; i >= 0; i--)
	{
		Layer* l = n->Layers[i];
		Backward_LayerGPU(l);
	}
}

Tensor* Seq_ForwardGPU(Model* n, Tensor* x)
{
	Tensor* y = Forward_LayerGPU(n->Layers[0]);
	for (int i = 1; i < n->n_layers; i++)
	{
		y = Forward_LayerGPU(n->Layers[i]);
	}
	return y;
}
void Seq_BackwardGPU(Model* n, Tensor* y)
{
	int N = n->n_layers;
	for (int i = N - 1; i >= 0; i--)
	{
		Backward_LayerGPU(n->Layers[i]);
	}
}
#endif // __NVCC__