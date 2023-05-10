#include "Net.h"
#include <stdlib.h>

void Net_Init(Net* net, Tensor* (*forward)(Net* n, Tensor* x, int is_training), float (*backward) (Net* n, Tensor* y), void (*init) (shape in))
{
	net->NetInit = init;
	net->NetForward = forward;
	net->NetBackward = backward;
}

float Backward_Layer(Layer* l, Tensor* y) 
{
	float loss = 0.f;
	switch (l->type)
	{
	case LT_DENSE: loss = Dense_Backward(l, y); break;
	case LT_SOFTMAX: loss = Softmax_Backward(l, y); break;
	case LT_RELU: loss = Relu_Backward(l, y); break;
	case LT_REGRESSION: loss = Regression_Backward(l, y); break;
	default:
		break;
	}
	return loss;
}

Tensor *Forward_Layer(Layer* l, Tensor* x)
{
	Tensor* y = NULL;
	switch (l->type)
	{
	case LT_INPUT: y = Input_Forward(l, x, 0); break;
	case LT_DENSE: y = Dense_Forward(l, x, 0); break;
	case LT_SOFTMAX: y = Softmax_Forward(l, x, 0); break;
	case LT_RELU: y = Relu_Forward(l, x, 0); break;
	case LT_REGRESSION: y = Regression_Forward(l, x, 0); break;
	default: break;
	}
	return y;
}

dList Net_getGradients(Net* n) 
{
	dList grads = dList_create();
	for (int i = 0; i < n->n_layers; i++)
	{
		Layer* l = n->Layers[i];
		switch (l->type)
		{
			case LT_DENSE: 
				Dense_GetGrads((Dense*)l->aData, &grads);
				break;
			default: break;
		}
	}
	return grads;
}

cJSON* Layer_To_JSON(Layer* l) 
{
	cJSON* Layer = cJSON_CreateObject();
	cJSON_AddItemToObject(Layer, "os", Shape_To_JSON(l->out_shape));
	cJSON_AddNumberToObject(Layer, "ni", l->n_inputs);
	cJSON_AddNumberToObject(Layer, "lt", l->type);
	cJSON_AddItemReferenceToObject(Layer, "o", Tensor_To_JSON(l->output));

	switch (l->type)
	{
		case LT_DENSE: {
			Dense* data = (Dense*)l->aData;
			cJSON_AddItemReferenceToObject(Layer, "d", Dense_To_JSON(data));
		}break;
		default:
			break;
	}

	return Layer;
}

void Layer_Load_JSON(Layer* t, cJSON* node)
{
	cJSON* output_shape = cJSON_GetObjectItem(node, "os"); //shape
	cJSON* layer_type = cJSON_GetObjectItem(node, "lt"); //type
	cJSON* output = cJSON_GetObjectItem(node, "o"); //tensor
	cJSON* num_inputs = cJSON_GetObjectItem(node, "ni"); //num_inputs
	cJSON* jData = cJSON_GetObjectItem(node, "d"); //Layer additional data

	shape os = (shape){ cJSON_GetArrayItem(output_shape, 0)->valueint,cJSON_GetArrayItem(output_shape, 1)->valueint,cJSON_GetArrayItem(output_shape, 2)->valueint };
	t->out_shape = os;
	t->n_inputs = num_inputs->valueint;
	t->type = (LayerType)layer_type->valueint;

	Tensor_Load_JSON(t->output, output);
	if(!cJSON_IsNull(jData))
		//Load layer data
		switch (t->type)
		{
			case LT_DENSE: {
				Dense* data = (Dense*)t->aData;
				Dense_Load_JSON(data, jData);
			}break;
			default:
				break;
		}
}

cJSON* Net_To_JSON(Net* n) 
{
	cJSON* jNet = cJSON_CreateObject();
	cJSON* jLayers = cJSON_CreateArray();
	cJSON_AddNumberToObject(jNet, "n_layers", n->n_layers);
	for (int i = 0; i < n->n_layers; i++)
	{
		cJSON* jLayer = Layer_To_JSON(n->Layers[i]);
		cJSON_AddItemReferenceToArray(jLayers, jLayer);
	}
	cJSON_AddItemToObject(jNet, "Layers", jLayers);
	return jNet;
}

void Net_Load_JSON(Net* t, cJSON* node) 
{
	cJSON* Layers = cJSON_GetObjectItem(node, "Layers"); //Layers
	cJSON* n_layers = cJSON_GetObjectItem(node, "n_layers"); //num_layers
	int n = n_layers->valueint;
	for (int i = 0; i < n; i++)
	{
		cJSON* l = cJSON_GetArrayItem(Layers, i);
		Layer_Load_JSON(t->Layers[i], l);
	}
}

