#include "Regression.h"
#include <stdlib.h>
#include <math.h>

Layer* Regression_Create(shape in_shape)
{
	Layer* dl = (Layer*)malloc(sizeof(Layer));

	dl->type = LT_REGRESSION;
	dl->n_inputs = in_shape.w * in_shape.h * in_shape.d;
	dl->out_shape = (shape){ 1, 1, dl->n_inputs };
	dl->output = Tensor_Create(dl->out_shape, 0, 0);

	dl->aData = NULL;
	return dl;
}

Tensor *Regression_Forward(Layer* l, Tensor* x, int is_train)
{
	l->input = x;
	Tensor_Copy(l->output, x);
	return l->output;
}


float Regression_Backward(Layer* l, Tensor* y)
{
	float loss = 0.f;
	Tensor* x = l->input;
	for (int i = 0; i < x->n; i++)
	{
		x->dw[i] = 0.f;
	}
	//tensor vector [0] = id [1] = value
	int i = (int)y->w[0];
	float val = y->w[1];
	float dy = x->w[i] - val;
	x->dw[i] = dy;
	loss += 0.5 * dy * dy;
	return loss;
}