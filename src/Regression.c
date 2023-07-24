#include "Regression.h"

Layer* Regression_Create(Layer* in)
{
	Layer* dl = malloc(sizeof(Layer));
	if (!dl)
	{
		printf("Regression allocation error!");
		return NULL;
	}
	dl->type = LT_REGRESSION;
	dl->n_inputs = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	dl->out_shape = (shape){ 1, 1, dl->n_inputs };
	dl->output = Tensor_Create(dl->out_shape, 0);
	dl->input = &in->output;
	LData* ld = (LData*)malloc(sizeof(LData));
	if (ld) {
		ld->loss = 0;
	}
	else printf("Regression data allocation error\n");
	dl->aData = ld;
	printf("Regression, output shape: [%d, %d, %d]\n", dl->out_shape.w, dl->out_shape.h, dl->out_shape.d);
	return dl;
}

Tensor *Regression_Forward(Layer* l)
{
	Tensor_CopyData(&l->output, l->input);
	return &l->output;
}

void Regression_Backward(Layer* l, Tensor* y)
{
	Tensor* x = l->input;
	int i = (int)y->w[0];
	float val = y->w[1];
	float dy = x->w[i] - val;
	x->dw[i] += dy;

	float dy2 = dy * dy;
	float loss = 0.5f * dy2;
	LData* ld = (LData*)l->aData;
	ld->loss = loss;
}