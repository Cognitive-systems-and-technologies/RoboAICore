#include "Softmax.h"
#include <stdlib.h>
#include <math.h>

Layer* Softmax_Create(Layer *in)
{
	Layer* dl = malloc(sizeof(Layer));
	Softmax* l = malloc(sizeof(Softmax));
	if (!dl)
	{
		printf("Softmax allocation error!");
		return NULL;
	}
	if (!l)
	{
		printf("Softmax data allocation error!");
		free(dl);
		return NULL;
	}
	dl->input = in;
	dl->type = LT_SOFTMAX;
	dl->n_inputs = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	dl->out_shape = (shape){ 1, 1, dl->n_inputs };
	dl->output = Tensor_Create(dl->out_shape, 0);

	l->sums = malloc(dl->out_shape.d*sizeof(float));
	if (!l->sums)
	{
		printf("Softmax es allocation error!");
		free(l);
		free(dl);
		return NULL;
	}
	for (int i = 0; i < dl->out_shape.d; i++)
	{
		l->sums[i] = 0.f;
	}
	dl->aData = l;
	return dl;
}

Tensor *Softmax_Forward(Layer* l)
{
	Softmax* data = (Softmax*)l->aData;
	Tensor* x = l->input;
	//get max
	float amax = x->w[0];
	for (int i = 1; i < l->out_shape.d; i++)
	{
		if (x->w[i] > amax)
			amax = x->w[i];
	}
	// compute exponentials (carefully to not blow up)
	float esum = 0.0f;
	for (int i = 0; i < l->out_shape.d; i++)
	{
		float e = (float)exp(x->w[i] - amax);
		esum += e;
		data->sums[i] = e;
	}
	// normalize output sum to one
	for (int i = 0; i < l->out_shape.d; i++)
	{
		data->sums[i] /= esum;
		l->output.w[i] = data->sums[i];
	}
	return &l->output;
}

void Softmax_Backward(Layer* l, Tensor* y)
{
	Softmax* data = l->aData;
	float loss = 0.f;
	Tensor* x = l->input;
	for (int i = 0; i < l->out_shape.d; i++)
	{
		float mul = -(y->w[i] - data->sums[i]);
		x->dw[i] += mul;
		if (y->w[i] > 0)
			loss += -(float)log(data->sums[i]);
	}
}