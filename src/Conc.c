#include "Conc.h"
#include <stdlib.h>

Layer* Conc_Create(Layer* in1, Layer *in2)
{
	Layer* l = (Layer*)malloc(sizeof(Layer));
	if (!l)
	{
		printf("Concatenate allocation error!");
		return NULL;
	}
	l->type = LT_CONC;
	//common layer def
	l->out_shape = (shape){ in1->out_shape.w, in1->out_shape.h, in1->out_shape.d+in2->out_shape.d };
	l->n_inputs = l->out_shape.w* l->out_shape.h* l->out_shape.d;
	l->output = Tensor_Create(l->out_shape, 0);
	l->input = &in1->output;
	l->input2 = &in2->output;

	l->aData = NULL;
	printf("Conc, output shape: [%d, %d, %d]\n", l->out_shape.w, l->out_shape.h, l->out_shape.d);
	return l;
}

Tensor* Conc_Forward(Layer* l)
{
	Tensor* t1 = l->input;
	Tensor* t2 = l->input2;

	shape s = l->out_shape;
	for (size_t d = 0; d < s.d; d++)
	{
		for (size_t h = 0; h < s.h; h++)
		{
			for (size_t w = 0; w < s.w; w++)
			{
				float val = (d < t1->s.d) ? Tensor_Get(t1, w, h, d) : Tensor_Get(t2, w, h, d - t1->s.d);
				Tensor_Set(&l->output, w, h, d, val);
			}
		}
	}
	return &l->output;
}

void Conc_Backward(Layer* l)
{
	Tensor* t1 = l->input;
	Tensor* t2 = l->input2;

	shape s = l->out_shape;
	for (size_t d = 0; d < s.d; d++)
	{
		for (size_t h = 0; h < s.h; h++)
		{
			for (size_t w = 0; w < s.w; w++)
			{
				if(d<t1->s.d)
				{
					int idx = tIdx(s, w, h, d);
					int ti = tIdx(t1->s, w, h, d);
					t1->dw[ti] = l->output.dw[idx];
				}
				else 
				{
					int idx = tIdx(s, w, h, d);
					int ti = tIdx(t2->s, w, h, d-t1->s.d);
					t2->dw[ti] = l->output.dw[idx];
				}
				//float grad = (d < t1->s.d) ? Tensor_Get(t1, w, h, d) : Tensor_Get(t2, w, h, d - t1->s.d);
			}
		}
	}
}

void Conc_BackpropGrads(Layer* l, Tensor *t1, Tensor *t2)
{
	shape s = l->out_shape;
	for (size_t d = 0; d < s.d; d++)
	{
		for (size_t h = 0; h < s.h; h++)
		{
			for (size_t w = 0; w < s.w; w++)
			{
				if (d < t1->s.d)
				{
					int idx = tIdx(s, w, h, d);
					int ti = tIdx(t1->s, w, h, d);
					t1->dw[ti] = l->output.dw[idx];
				}
				else
				{
					int idx = tIdx(s, w, h, d);
					int ti = tIdx(t2->s, w, h, d - t1->s.d);
					t2->dw[ti] = l->output.dw[idx];
				}
				//float grad = (d < t1->s.d) ? Tensor_Get(t1, w, h, d) : Tensor_Get(t2, w, h, d - t1->s.d);
			}
		}
	}
}