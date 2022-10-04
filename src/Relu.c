#include "Relu.h"
#include <stdlib.h>

Layer* Relu_Create(shape out_shape)
{
	Layer* dl = (Layer*)malloc(sizeof(Layer));
	dl->type = LT_RELU;
	dl->aData = NULL;
	dl->n_inputs = out_shape.w * out_shape.h * out_shape.d;
	dl->out_shape = out_shape;
	dl->output = Vol_Create(dl->out_shape, 0, 0);
	return dl;
}

Vol *Relu_Forward(Layer* l, Vol* x, int is_train)
{
	l->input = x;
	for (int i = 0; i < x->n; i++)
	{
		if (x->w[i] < 0) l->output->w[i] = 0;
		else
			l->output->w[i] = x->w[i];
	}
	return l->output;
}

float Relu_Backward(Layer* l, Vol* y)
{
	float loss = 0.f;

	Vol* x = l->input;
	for (int i = 0; i < x->n; i++)
	{
		x->dw[i] = 0.f;
	}
	//---------------------------------------
	for (int i = 0; i < x->n; i++)
	{
		if (l->output->w[i] <= 0) x->dw[i] = 0; // threshold
		else x->dw[i] = l->output->dw[i];
	}
	return loss;
}