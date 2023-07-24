#include "Dense.h"
#include <stdlib.h>

Layer* Dense_Create(int num_neurons, RandType weightInit, Layer* in)
{
	Layer* l = (Layer*)malloc(sizeof(Layer));
	if (!l)
	{
		printf("Dense allocation error!");
		return NULL;
	}
	l->type = LT_DENSE;
	int inn = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	//common layer def
	l->out_shape = (shape){ 1, 1, num_neurons };
	l->n_inputs = inn;
	l->output = Tensor_Create(l->out_shape, 0);
	l->input = &in->output;

	float bias = 0.0f;

	Dense *ld = (Dense*)malloc(sizeof(Dense));
	if (ld) {
		//ld->activation = act;
		ld->n = num_neurons;
		ld->kernels = (Tensor*)malloc(sizeof(Tensor) * num_neurons);
		if (ld->kernels) {
			shape kernels_shape = { 1, 1, inn };//each row is weight
			for (size_t i = 0; i < num_neurons; i++)
			{
				ld->kernels[i] = Tensor_Create(kernels_shape, 0.f);

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
			ld->biases = Tensor_Create((shape){ 1, 1, num_neurons }, bias);
		}
		else printf("Kernels allocation error\n");
	}
	else printf("Dense data allocation error\n");
	l->aData = ld;
	printf("Dense, output shape: [%d, %d, %d]\n", l->out_shape.w, l->out_shape.h, l->out_shape.d);
	return l;
}

Tensor* Dense_Forward(Layer* l) 
{
	Tensor* x = l->input;
	Dense* data = (Dense*)l->aData;
	for (int d = 0; d < l->out_shape.d; d++) //foreach kernel
	{
		float wsum = 0;
		for (int i = 0; i < x->n; i++)
		{
			//int id = tIdx(data->kernels.s, 0, i, d);
			//float wi = Tensor_Get(&data->kernels, 0, i, d);
			wsum += x->w[i] * data->kernels[d].w[i];
		}
		wsum += data->biases.w[d];

		//if (data->activation == A_RELU)
		//	if (wsum < 0) wsum = 0.0001f * wsum;

		l->output.w[d] = wsum;
	}
	return &l->output;
}

void Dense_Backward(Layer* l) 
{
	Dense* data = (Dense*)l->aData;
	Tensor* x = l->input;
	for (int d = 0; d < l->out_shape.d; d++)
	{
		float next_grad = l->output.dw[d];
		//if(data->activation == A_RELU)
		//	if (l->output.w[d] < 0) chain_grad = 0.0001f * chain_grad; // threshold
		for (int h = 0; h < l->n_inputs; h++)
		{
			//int idx = tIdx(ke->s, 0, h, d);
			x->dw[h] += data->kernels[d].w[h] * next_grad; // grad wrt input data
			data->kernels[d].dw[h] += x->w[h] * next_grad; // grad wrt params
		}
		data->biases.dw[d] += next_grad;
	}
}

cJSON* Dense_To_JSON(Dense* d)
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

void Dense_Load_JSON(Dense* d, cJSON* node)
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
	}*/
}

void Dense_Free(Layer* l) 
{
	Dense* data = (Dense*)l->aData;
	for (int i = 0; i < data->n; i++)
	{
		Tensor_Free(&data->kernels[i]);
	}
	Tensor_Free(&data->biases);
	free(data);
	Tensor_Free(&l->output);
	free(l);
}