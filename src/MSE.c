#include "MSE.h"

Layer* MSE_Create(shape in_shape)
{
	Layer* dl = malloc(sizeof(Layer));
	if (!dl)
	{
		printf("MSE allocation error!");
		return NULL;
	}
	dl->type = LT_MSE;
	dl->n_inputs = in_shape.w * in_shape.h * in_shape.d;
	dl->out_shape = (shape){ 1, 1, dl->n_inputs };
	dl->output = Tensor_Create(dl->out_shape, 0, 0);
	dl->input = NULL;
	dl->aData = NULL;
	return dl;
}

Tensor * MSE_Forward(Layer* l, Tensor* x, int is_train)
{
	l->input = x;
	Tensor_Copy(l->output, x);
	return l->output;
}

float MSE_Backward(Layer* l, Tensor* y_true)
{
	Tensor* x = l->input;
	/*for (int i = 0; i < x->n; i++)
	{
		x->dw[i] = 0.f;
	}*/
	memset(x->dw, 0, sizeof(float)*x->n);
	
	float sum = 0;
	for (size_t i = 0; i < x->n; i++)
	{
		float dy = (2.f/x->n) * (x->w[i] - y_true->w[i]);
		x->dw[i] = dy;

		float t = y_true->w[i] - x->w[i];
		sum += t*t;
	}

	float loss = sum / x->n;
	return loss;
}