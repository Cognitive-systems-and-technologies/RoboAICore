#include "Net.h"
#include <stdlib.h>

void Net_Init(Net* net, Tensor* (*forward)(Net* n, Tensor* x, int is_training), float (*backward) (Net* n, Tensor* y), void (*init) (shape in))
{
	net->NetInit = init;
	net->NetForward = forward;
	net->NetBackward = backward;
}

float Backward_Layer(Layer* l, Tensor* y) 
{
	float loss = 0.f;
	switch (l->type)
	{
	case LT_DENSE: loss = Dense_Backward(l, y); break;
	case LT_SOFTMAX: loss = Softmax_Backward(l, y); break;
	case LT_RELU: loss = Relu_Backward(l, y); break;
	default:
		break;
	}
	return loss;
}

Tensor *Forward_Layer(Layer* l, Tensor* x)
{
	Tensor* y = NULL;
	switch (l->type)
	{
	case LT_INPUT: y = Input_Forward(l, x, 0); break;
	case LT_DENSE: y = Dense_Forward(l, x, 0); break;
	case LT_SOFTMAX: y = Softmax_Forward(l, x, 0); break;
	case LT_RELU: y = Relu_Forward(l, x, 0); break;
	default: break;
	}
	return y;
}