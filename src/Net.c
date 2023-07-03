#include "Net.h"
#include <stdlib.h>

Net Net_Create() 
{
	Net n;
	n.Layers = NULL;
	n.n_layers = 0;
	n.NetForward = Seq_Forward;
	n.NetBackward = Seq_Backward;
	return n;
}

Layer* Net_AddLayer(Net* n, Layer* l) 
{
	int cnt = n->n_layers + 1;
	Layer** tmp = (Layer**)realloc(n->Layers, sizeof(Layer*) * cnt);
	if (!tmp) {
		free(n->Layers);
		n->Layers = NULL;
		return NULL;
	}
	n->n_layers = cnt;
	n->Layers = tmp;
	n->Layers[cnt - 1] = l;
	return n->Layers[cnt - 1];
}

void Backward_Layer(Layer* l, Tensor* y) 
{
	switch (l->type)
	{
	case LT_DENSE: Dense_Backward(l); break;
	case LT_SOFTMAX: break;
	case LT_RELU: break;
	case LT_REGRESSION: break;
	case LT_MSE: MSE_Backward(l,y); break;
	case LT_TANHA: TanhA_Backward(l); break;
	default:
		break;
	}
}

Tensor *Forward_Layer(Layer* l, Tensor* x)
{
	Tensor* y = NULL;
	switch (l->type)
	{
	case LT_INPUT: y = Input_Forward(l, x); break;
	case LT_DENSE: y = Dense_Forward(l); break;
	case LT_SOFTMAX: break;
	case LT_RELU: y = Relu_Forward(l); break;
	case LT_REGRESSION: break;
	case LT_MSE: y = MSE_Forward(l); break;
	case LT_TANHA: y = TanhA_Forward(l); break;
	case LT_CONV: y = Conv2d_Forward(l); break;
	case LT_MAXPOOL: y = MaxPool2d_Forward(l); break;
	default: break;
	}
	return y;
}

Tensor* Seq_Forward(Net* n, Tensor* x)
{
	Tensor* y = Forward_Layer(n->Layers[0], x);
	for (int i = 1; i < n->n_layers; i++)
	{
		y = Forward_Layer(n->Layers[i], y);
	}
	return y;
}

void Seq_Backward(Net* n, Tensor* y)
{
	int N = n->n_layers;
	for (int i = N - 1; i >= 0; i--)
	{
		Layer* l = n->Layers[i];
		switch (l->type)
		{
		case LT_DENSE: Dense_Backward(l); break;
		case LT_SOFTMAX: break;
		case LT_RELU: Relu_Backward(l); break;
		case LT_REGRESSION: break;
		case LT_MSE: MSE_Backward(l, y); break;
		case LT_TANHA: TanhA_Backward(l); break;
		case LT_CONV: Conv2d_Backward(l); break;
		case LT_MAXPOOL: MaxPool2d_Backward(l); break;
		}
	}
}