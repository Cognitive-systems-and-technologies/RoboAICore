#include "MSE.h"

Layer* MSE_Create(Layer *in)
{
	Layer* l = (Layer*)malloc(sizeof(Layer));
	if (!l)
	{
		printf("MSE allocation error!");
		return NULL;
	}
	l->type = LT_MSE;
	l->n_inputs = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	l->out_shape = (shape){ 1, 1, l->n_inputs };
	l->output = Tensor_Create(l->out_shape, 0);
	l->input = &in->output;

	LData* ld = (LData*)malloc(sizeof(LData));
	if (ld) {
		ld->loss = 0;
	}
	else printf("MSE data allocation error\n");
	l->aData = ld;
	printf("Mse, output shape: [%d, %d, %d]\n", l->out_shape.w, l->out_shape.h, l->out_shape.d);
	return l;
}

Tensor * MSE_Forward(Layer* l)
{
	Tensor_CopyData(&l->output, l->input);
	return &l->output;
}

void MSE_Backward(Layer* l, Tensor* y_true)
{
	Tensor* x = l->input;
	float sum = 0;
	for (int i = 0; i < x->n; i++)
	{
		float dy = (2.f/(float)x->n) * (x->w[i] - y_true->w[i]);
		x->dw[i] += dy;

		float t = y_true->w[i] - x->w[i];
		sum += t*t;
	}
	float loss = sum / (float)x->n;
	LData* ld = (LData*)l->aData;
	ld->loss = loss;
}