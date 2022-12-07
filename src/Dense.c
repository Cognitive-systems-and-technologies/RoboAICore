#include "Dense.h"
#include <stdlib.h>

Layer* Dense_Create(int num_neurons, shape in_shape) 
{
	Layer* dl = malloc(sizeof(Layer));
	if (!dl)
	{
		printf("Dense allocation error!");
		return NULL;
	}
	Dense* l = malloc(sizeof(Dense));
	if (!l)
	{
		printf("Dense data allocation error!");
		free(dl);
		return NULL;
	}
	dl->type = LT_DENSE;
	//common layer def
	dl->out_shape = (shape){ 1, 1, num_neurons };
	dl->n_inputs = in_shape.w * in_shape.h * in_shape.d;
	dl->output = Tensor_Create(dl->out_shape, 0, 0);
	dl->input = NULL;
	// optional
	l->l1_decay_mul = 0.0f;
	l->l2_decay_mul = 1.0f;

	float bias = 0.0f;

	l->n_filters = dl->out_shape.d;
	l->filters = malloc(sizeof(Tensor)*dl->out_shape.d);
	if (!l->filters)
	{
		printf("Dense filters allocation error!");
		free(l);
		free(dl);
		return NULL;
	}
	for (int i = 0; i < dl->out_shape.d; i++)
	{
		const float r = (float)rand() / (float)(RAND_MAX / 1.f);
		Tensor_Init(&l->filters[i], (shape) { 1, 1, dl->n_inputs }, r, 1);
	}
	l->biases = Tensor_Create((shape) { 1, 1, dl->out_shape.d }, bias, 1);
	
	dl->aData = l;
	return dl;
}

Tensor *Dense_Forward(Layer* l, Tensor* x, int is_train) 
{
	Dense* data = (Dense*)l->aData;
	l->input = x; //save pointer to previous layer output
	for (int i = 0; i < l->out_shape.d; i++) //foreach output neuron
	{
		float a = Tensor_WeightedSum(x, &data->filters[i]);
		a += data->biases->w[i];//add bias
		l->output->w[i] = a;
	}
	return l->output;
}

float Dense_Backward(Layer* l, Tensor* y)
{
	Dense* data = l->aData;
	float loss = 0.f;

	Tensor* x = l->input;
	for (int i = 0; i < x->n; i++)
	{
		x->dw[i] = 0.f;
	}
	//---------------------------------------
	for (int i = 0; i < l->out_shape.d; i++)
	{
		Tensor tfi = data->filters[i];
		float chain_grad = l->output->dw[i];
		for (int d = 0; d < l->n_inputs; d++)
		{
			x->dw[d] += tfi.w[d] * chain_grad; // grad wrt input data
			tfi.dw[d] += x->w[d] * chain_grad; // grad wrt params
		}
		data->biases->dw[i] += chain_grad;
	}
	return loss;
}

void Dense_Free(Dense* l) 
{
	//Tensor_Free(l->output);
	Tensor_Free(l->biases);
	Tensor_Free(l->filters);
	free(l);
}

cJSON* Dense_To_JSON(Dense* d)
{
	cJSON* Data = cJSON_CreateObject();
	cJSON* fi = cJSON_CreateArray();

	cJSON_AddNumberToObject(Data, "l1", d->l1_decay_mul);
	cJSON_AddNumberToObject(Data, "l2", d->l2_decay_mul);
	cJSON_AddNumberToObject(Data, "nf", d->n_filters);

	for (int i = 0; i < d->n_filters; i++)
	{
		cJSON_AddItemToArray(fi, Tensor_To_JSON(&d->filters[i]));
		//cJSON_AddItemToObject();
	}
	cJSON_AddItemToObject(Data, "filters", fi);
	cJSON_AddItemReferenceToObject(Data, "biases", Tensor_To_JSON(d->biases));

	return Data;
}

void Dense_Load_JSON(Dense* d, cJSON* node) 
{
	cJSON* l1 = cJSON_GetObjectItem(node, "l1");
	cJSON* l2 = cJSON_GetObjectItem(node, "l2");
	cJSON* nf = cJSON_GetObjectItem(node, "nf");

	cJSON* filters = cJSON_GetObjectItem(node, "filters");//array
	cJSON* biases = cJSON_GetObjectItem(node, "biases");

	d->l1_decay_mul = (float)l1->valuedouble;
	d->l2_decay_mul = (float)l2->valuedouble;
	//load biases
	Tensor_Load_JSON(d->biases, biases);
	//load filters
	int n = nf->valueint;
	for (int i = 0; i < n; i++)
	{
		cJSON* f = cJSON_GetArrayItem(filters, i);
		Tensor_Load_JSON(&d->filters[i], f);
	}
}