#include "Net.h"
#include <stdlib.h>

void Net_Init(Net* net, Vol* (*forward)(Net* n, Vol* x, int is_training), float (*backward) (Net* n, Vol* y), void (*init) (shape in))
{
	net->NetInit = init;
	net->NetForward = forward;
	net->NetBackward = backward;
}

float Backward_Layer(Layer* l, Vol* y) 
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

Vol *Forward_Layer(Layer* l, Vol* x)
{
	Vol* y = NULL;
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