#include "Input.h"

#ifdef __NVCC__
Layer* Input_CreateGPU(shape out_shape)
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
	dl->out_shape = { out_shape.w, out_shape.h, out_shape.d };
	dl->output = Tensor_CreateGPU(dl->out_shape, 0);
	printf("Input GPU, output shape: [%d, %d, %d]\n", dl->out_shape.w, dl->out_shape.h, dl->out_shape.d);
	return dl;
}

Tensor* Input_ForwardGPU(Layer* l, Tensor* x) 
{
	Tensor_CopyDataGPU(&l->output, x);
	return &l->output;
}
#endif // __NVCC__
