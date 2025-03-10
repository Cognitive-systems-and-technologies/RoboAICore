#include "Input.h"

Layer* Input_Create(shape out_shape)
{
	Layer* dl = (Layer*)malloc(sizeof(Layer));
	if (!dl)
	{
		printf("Input allocation error!");
		return NULL;
	}
	dl->input = NULL;
	dl->type = LT_INPUT;
	dl->aData = NULL;
	dl->n_inputs = out_shape.w * out_shape.h * out_shape.d;
	dl->out_shape = (shape){ out_shape.w, out_shape.h, out_shape.d };
	dl->output = Tensor_Create(dl->out_shape, 0);
	printf("Input, output shape: [%d, %d, %d]\n", dl->out_shape.w, dl->out_shape.h, dl->out_shape.d);

	return dl;
}

Tensor *Input_Forward(Layer* l)
{
	Tensor_CopyData(&l->output, l->input);
	return &l->output;
}

void Input_Backward(Layer* l)
{
	Tensor* x = l->input;
	//for (int i = 0; i < l->output.n; i++)
	//{
	//	x->dw[i] += l->output.dw[i];
	//}
	memcpy(x->dw, l->output.dw, sizeof(float) * l->output.n);
}

void Input_Free(Layer* l) 
{
	Tensor_Free(&l->output);
	free(l);
}