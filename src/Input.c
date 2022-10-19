#include "Input.h"
#include <stdlib.h>

Layer* Input_Create(shape out_shape)
{
	Layer* dl = (Layer*)malloc(sizeof(Layer));
	dl->type = LT_INPUT;
	dl->aData = NULL;
	dl->n_inputs = out_shape.w * out_shape.h * out_shape.d;
	dl->out_shape = out_shape;
	dl->output = Tensor_Create(dl->out_shape, 0, 0);
	return dl;
}

Tensor *Input_Forward(Layer* l, Tensor* x, int is_train)
{
	l->input = x;
	Tensor_Copy(l->output, x);
	return l->output;
}