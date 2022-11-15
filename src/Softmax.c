#include "Softmax.h"
#include <stdlib.h>
#include <math.h>

Layer* Softmax_Create(shape in_shape)
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
	dl->type = LT_SOFTMAX;
	dl->n_inputs = in_shape.w * in_shape.h * in_shape.d;
	dl->out_shape = (shape){ 1, 1, dl->n_inputs };
	dl->output = Tensor_Create(dl->out_shape, 0, 0);

	l->es = malloc(dl->out_shape.d*sizeof(float));
	if (!l->es)
	{
		printf("Softmax es allocation error!");
		free(l);
		free(dl);
		return NULL;
	}
	for (int i = 0; i < dl->out_shape.d; i++)
	{
		l->es[i] = 0.f;
	}
	dl->aData = l;
	return dl;
}

Tensor *Softmax_Forward(Layer* l, Tensor* x, int is_train)
{
	l->input = x; //save pointer to previous layer output
	Softmax* data = (Softmax*)l->aData;

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
		data->es[i] = e;
	}
	// normalize output sum to one
	for (int i = 0; i < l->out_shape.d; i++)
	{
		data->es[i] /= esum;
		l->output->w[i] = data->es[i];
	}
	return l->output;
}


float Softmax_Backward(Layer* l, Tensor* y)
{
	Softmax* data = l->aData;
	float loss = 0.f;
	Tensor* x = l->input;
	for (int i = 0; i < x->n; i++)
	{
		x->dw[i] = 0.f;
	}
	for (int i = 0; i < l->out_shape.d; i++)
	{
		float mul = -(y->w[i] - data->es[i]);
		x->dw[i] = mul;
		if (y->w[i] > 0)
			loss = -(float)log(data->es[i]);
	}
	return loss;
}