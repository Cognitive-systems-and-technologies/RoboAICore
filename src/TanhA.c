#include "TanhA.h"
#include <stdlib.h>
#include <math.h>

Layer* TanhA_Create(Layer* in)
{
	Layer* dl = (Layer*)malloc(sizeof(Layer));
	if (!dl)
	{
		printf("Tanh allocation error!");
		return NULL;
	}
	dl->type = LT_TANHA;
	dl->n_inputs = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	dl->out_shape = (shape){ in->out_shape.w, in->out_shape.h, in->out_shape.d };
	dl->output = Tensor_Create(dl->out_shape, 0);
	dl->input = &in->output;
	dl->aData = NULL;
	printf("Tanh activation, output shape: [%d, %d, %d]\n", dl->out_shape.w, dl->out_shape.h, dl->out_shape.d);

	return dl;
}

Tensor* TanhA_Forward(Layer* l) 
{
	Tensor* y = &l->output;
	for (int i = 0; i < l->input->n; i++)
	{
		y->w[i] = tanhf(l->input->w[i]);
	}
	return y;
}

void TanhA_Backward(Layer* l) 
{
	Tensor* x = l->input;
	Tensor* out = &l->output;

	for (size_t i = 0; i < x->n; i++)
	{
		float xwi = out->w[i];
		x->dw[i] += (1.f - xwi * xwi) * out->dw[i];//mult by chain gradient
	}
}