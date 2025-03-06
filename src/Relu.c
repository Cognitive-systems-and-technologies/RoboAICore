#include "Relu.h"
#include <stdlib.h>

Layer* Relu_Create(Layer* in)
{
	Layer* dl = (Layer*)malloc(sizeof(Layer));
	if (!dl)
	{
		printf("Relu allocation error!");
		return NULL;
	}
	dl->type = LT_RELU;
	dl->aData = NULL;
	dl->n_inputs = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	dl->out_shape = (shape){ in->out_shape.w, in->out_shape.h, in->out_shape.d };
	dl->output = Tensor_Create(dl->out_shape, 0);
	dl->input = &in->output;
	printf("Relu, output shape: [%d, %d, %d]\n", dl->out_shape.w, dl->out_shape.h, dl->out_shape.d);
	return dl;
}

Tensor* Relu_Forward(Layer* l)
{
	Tensor* x = l->input;
	for (int i = 0; i < x->n; i++)
	{
		if (x->w[i] < 0) l->output.w[i] = 0.0;
		else
			l->output.w[i] = x->w[i];
	}
	return &l->output;
}

void Relu_Backward(Layer* l)
{
	Tensor* x = l->input;
	for (int i = 0; i < x->n; i++)
	{
		if (x->w[i] < 0) x->dw[i] += 0.0; // threshold
		else x->dw[i] += l->output.dw[i];
	}
}