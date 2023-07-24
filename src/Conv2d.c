#include "Conv2d.h"
#include <stdlib.h>

Layer* Conv2d_Create(int num_kernels, shape2 k_size, shape2 stride, int pad, RandType weightInit, Layer* in)
{
	//input shape depth must be == 1
	Layer* l = (Layer*)malloc(sizeof(Layer));
	if (!l)
	{
		printf("Conv2d allocation error!");
		return NULL;
	}
	l->type = LT_CONV;
	int inn = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	//calculate output shape
	l->out_shape.d = num_kernels;
	l->out_shape.w = (int)((in->out_shape.w - k_size.w + pad * 2) / stride.w + 1);
	l->out_shape.h = (int)((in->out_shape.h - k_size.h + pad * 2) / stride.h + 1);
	printf("Conv2d, output shape: [%d, %d, %d] pad: %d\n", l->out_shape.w, l->out_shape.h, l->out_shape.d, pad);

	l->n_inputs = inn;
	l->output = Tensor_Create(l->out_shape, 0);
	l->input = &in->output;

	float bias = 0.0f;
	Conv2d *ld = (Conv2d*)malloc(sizeof(Conv2d));
	if (ld) {
		ld->n = num_kernels;
		ld->pad = pad;
		ld->stride.w = stride.w; ld->stride.h = stride.h;
		ld->k_size.w = k_size.w; ld->k_size.h = k_size.h;

		//create kernels
		ld->kernels = (Tensor*)malloc(sizeof(Tensor) * num_kernels);
		if (ld->kernels) {
			shape ks = { k_size.w, k_size.h, in->out_shape.d };
			for (size_t i = 0; i < num_kernels; i++)
			{
				ld->kernels[i] = Tensor_Create(ks, 1.f); //assume that n input channels = 1 for now
				//Tensor_He_Rand(ld->kernels[i].w, ld->kernels[i].n);
				switch (weightInit)
				{
				case R_XAVIER:
					Tensor_Xavier_Rand(ld->kernels[i].w, ld->kernels[i].n);
					break;
				case R_HE:
					Tensor_He_Rand(ld->kernels[i].w, ld->kernels[i].n);
					break;
				default:
					break;
				}
			}
			ld->biases = Tensor_Create((shape){ 1, 1, num_kernels }, bias);
		}
	}
	else printf("Conv2d data allocation error\n");
	l->aData = ld;
	return l;
}

Tensor* Conv2d_Forward(Layer* l)
{
	Tensor* inp = l->input;
	Conv2d* data = (Conv2d*)l->aData;

	int pad = data->pad;
	for (size_t d = 0; d < l->out_shape.d; d++)
	{
		for (size_t h = 0; h < l->out_shape.h; h++)
		{
			for (size_t w = 0; w < l->out_shape.w; w++)
			{
				float ksum = 0;
				//iterate kernels by size
				for (size_t kh = 0; kh < data->k_size.h; kh++)
				{
					int cury = (h * data->stride.h - pad) + kh;
					for (size_t kw = 0; kw < data->k_size.w; kw++)
					{
						int curx = (w * data->stride.w - pad) + kw;
						//for image depth
						for (size_t imd = 0; imd < inp->s.d; imd++)
						{
							if (curx >= 0&& cury >=0&& curx < inp->s.w && cury < inp->s.h)
							{
								int imi = ((l->input->s.w * cury) + curx) * l->input->s.d + imd;
								int ki = ((data->kernels[d].s.w * kh) + kw) * data->kernels[d].s.d + imd; 
								ksum += data->kernels[d].w[ki] * inp->w[imi];
							}
						}
					}
				}
				ksum += data->biases.w[d];
				Tensor_Set(&l->output, w, h, d, ksum);
			}
		}
	}
	return &l->output;
}

void Conv2d_Backward(Layer* l)
{
	Tensor* inp = l->input;
	Conv2d* data = (Conv2d*)l->aData;

	int pad = data->pad;
	for (size_t d = 0; d < l->out_shape.d; d++)
	{
		for (size_t h = 0; h < l->out_shape.h; h++)
		{
			for (size_t w = 0; w < l->out_shape.w; w++)
			{
				int idx = tIdx(l->output.s, w, h, d);
				float next_grad = l->output.dw[idx];
				//iterate kernels by size
				for (size_t kh = 0; kh < data->k_size.h; kh++)
				{
					int cury = (h * data->stride.h - pad) + kh;
					for (size_t kw = 0; kw < data->k_size.w; kw++)
					{
						int curx = (w * data->stride.w - pad) + kw;
						for (size_t imd = 0; imd < inp->s.d; imd++)
						{
							if (curx >= 0 && cury >= 0 && curx < inp->s.w && cury < inp->s.h)
							{
								int imi = ((l->input->s.w * cury) + curx) * l->input->s.d + imd;
								int ki = ((data->kernels[d].s.w * kh) + kw) * data->kernels[d].s.d + imd;
								data->kernels[d].dw[ki] += inp->w[imi] * next_grad;
								inp->dw[imi] += data->kernels[d].w[ki] * next_grad;
							}
						}
					}
				}
				data->biases.dw[d] += next_grad;
			}
		}
	}
}

cJSON* Conv2d_To_JSON(Conv2d* d) 
{
	cJSON* Data = cJSON_CreateObject();
	cJSON* fi = cJSON_CreateArray();

	cJSON_AddNumberToObject(Data, "nk", d->biases.n);

	for (int i = 0; i < d->biases.n; i++)
	{
		cJSON_AddItemToArray(fi, Tensor_To_JSON(&d->kernels[i]));
	}
	cJSON_AddItemToObject(Data, "kernels", fi);
	cJSON_AddItemToObject(Data, "biases", Tensor_To_JSON(&d->biases));
	return Data;
}

void Conv2d_Load_JSON(Conv2d* d, cJSON* node)
{
	cJSON* nk = cJSON_GetObjectItem(node, "nk");
	cJSON* kernels = cJSON_GetObjectItem(node, "kernels");//array
	cJSON* biases = cJSON_GetObjectItem(node, "biases");
	//load biases
	Tensor_Load_JSON(&d->biases, biases);
	//load kernels
	int n = nk->valueint;
	int i = 0;
	cJSON* kernel = NULL;
	cJSON_ArrayForEach(kernel, kernels)
	{
		Tensor_Load_JSON(&d->kernels[i], kernel);
		i++;
	}
	/*
	for (int i = 0; i < n; i++)
	{
		cJSON* f = cJSON_GetArrayItem(kernels, i);
		Tensor_Load_JSON(&d->kernels[i], f);
	}
	*/
}

void Conv2d_Free(Layer* l) 
{
	Conv2d* data = (Conv2d*)l->aData;
	for (int i = 0; i < data->n; i++)
	{
		Tensor_Free(&data->kernels[i]);
	}
	Tensor_Free(&data->biases);
	free(data);
	Tensor_Free(&l->output);
	free(l);
}