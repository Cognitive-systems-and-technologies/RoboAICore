#include "TanhA.h"
#include <stdlib.h>
#include <math.h>

Layer* TanhA_Create(shape in_shape)
{
	Layer* dl = malloc(sizeof(Layer));
	if (!dl)
	{
		printf("Tanh allocation error!");
		return NULL;
	}
	dl->type = LT_TANHA;
	dl->n_inputs = in_shape.w * in_shape.h * in_shape.d;
	dl->out_shape = (shape){ in_shape.w, in_shape.h, in_shape.d };
	dl->output = Tensor_Create(dl->out_shape, 0, 0);
	dl->input = NULL;
	dl->aData = NULL;
	return dl;
}

Tensor * TanhA_Forward(Layer* l, Tensor* x, int is_train)
{
	l->input = x;
	Tensor* y = l->output;
	for (int i = 0; i < x->n; i++)
	{
		y->w[i] = tanhf(x->w[i]);
	}
	return y;
}

float TanhA_Backward(Layer* l, Tensor* y)
{
	Tensor* x = l->input;
	Tensor* out = l->output;
	/*for (int i = 0; i < x->n; i++)
	{
		x->dw[i] = 0.f;
	}*/
	memset(x->dw, 0, sizeof(float) * x->n);

	for (size_t i = 0; i < x->n; i++)
	{
		float xwi = x->w[i];
		x->dw[i] = (1.f - xwi * xwi) * out->dw[i];//mult by chain gradient
	}

	return 0;
}