#include "MaxPool2d.h"
#include <stdlib.h>

Layer* MaxPool2d_Create(shape2 k_size, shape2 stride, int pad, Layer* in)
{
	//input shape depth must be == 1
	Layer* l = (Layer*)malloc(sizeof(Layer));
	if (!l)
	{
		printf("MaxPool2d allocation error!");
		return NULL;
	}
	l->type = LT_MAXPOOL;
	int inn = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	//calculate output shape
	l->out_shape.d = in->out_shape.d;
	l->out_shape.w = (int)((in->out_shape.w - k_size.w + pad * 2) / stride.w + 1);
	l->out_shape.h = (int)((in->out_shape.h - k_size.h + pad * 2) / stride.h + 1);
	printf("MaxPool2d output shape: [%d, %d, %d]\n", l->out_shape.w, l->out_shape.h, l->out_shape.d);

	l->n_inputs = inn;
	l->output = Tensor_Create(l->out_shape, 0);
	l->input = &in->output;

	float bias = 0.0f;
	MaxPool2d*ld = (MaxPool2d*)malloc(sizeof(MaxPool2d));
	if (ld) {
		ld->pad = pad;
		ld->stride.w = stride.w; ld->stride.h = stride.h;
		ld->k_size.w = k_size.w; ld->k_size.h = k_size.h;
	}
	else printf("MaxPool2d data allocation error\n");
	l->aData = ld;
	return l;
}

Tensor* MaxPool2d_Forward(Layer* l)
{
	Tensor* inp = l->input;
	MaxPool2d* data = (MaxPool2d*)l->aData;

	int pad = data->pad;
	for (size_t d = 0; d < l->out_shape.d; d++)
	{
		for (size_t h = 0; h < l->out_shape.h; h++)
		{
			for (size_t w = 0; w < l->out_shape.w; w++)
			{
				float maxk = -FLT_MAX;
				//iterate kernels by size
				for (size_t kh = 0; kh < data->k_size.h; kh++)
				{
					int cury = (h * data->stride.h - pad) + kh;
					for (size_t kw = 0; kw < data->k_size.w; kw++)
					{
						int curx = (w * data->stride.w - pad) + kw;
						if (curx >= 0&& cury >=0&& curx < inp->s.w && cury < inp->s.h)
						{
							float xwi = Tensor_Get(inp, curx, cury, d);
							if (xwi > maxk) { maxk = xwi;}
						}
					}
				}
				Tensor_Set(&l->output, w, h, d, maxk);
			}
		}
	}
	return &l->output;
}

void MaxPool2d_Backward(Layer* l)
{
	Tensor* inp = l->input;
	MaxPool2d* data = (MaxPool2d*)l->aData;
	Tensor* out = &l->output;
	
	int pad = data->pad;
	for (size_t d = 0; d < l->out_shape.d; d++)
	{
		for (size_t h = 0; h < l->out_shape.h; h++)
		{
			for (size_t w = 0; w < l->out_shape.w; w++)
			{
				float maxk = -FLT_MAX;
				int khm=0, kwm=0;
				//iterate kernels by size
				for (size_t kh = 0; kh < data->k_size.h; kh++)
				{
					int cury = (h * data->stride.h - pad) + kh;
					for (size_t kw = 0; kw < data->k_size.w; kw++)
					{
						int curx = (w * data->stride.w - pad) + kw;
						if (curx >= 0 && cury >= 0 && curx < inp->s.w && cury < inp->s.h)
						{
							float xwi = Tensor_Get(inp, curx, cury, d);
							if (xwi > maxk) { maxk = xwi; kwm = curx; khm = cury; }
						}
					}
				}
				int ido = tIdx(out->s, w, h, d);
				float next_grad = out->dw[ido];
				int id = tIdx(inp->s, kwm, khm, d);
				inp->dw[id] += next_grad;
			}
		}
	}
}