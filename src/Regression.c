#include "Regression.h"
#include <stdlib.h>
#include <math.h>

Layer* Regression_Create(shape in_shape)
{
	Layer* dl = malloc(sizeof(Layer));
	if (!dl)
	{
		printf("Regression allocation error!");
		return NULL;
	}
	dl->type = LT_REGRESSION;
	dl->n_inputs = in_shape.w * in_shape.h * in_shape.d;
	dl->out_shape = (shape){ 1, 1, dl->n_inputs };
	dl->output = Tensor_Create(dl->out_shape, 0, 0);
	dl->input = NULL;
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


	//if ((p2 / dy) != dy) {
		//printf("ERROR_OVERFLOW");
	//	return 0;
	//}

	if ((dy < 0.0f) == (dy < 0.0f)
		&& abs(dy) > FLT_MAX - abs(dy)) {
		printf("ERROR_OVERFLOW");
		return 0;
	}
	else {
		float p2 = dy * dy;
		float loss = 0.5f * p2;
		return loss;
	}
}