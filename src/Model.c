#include "Model.h"
#include <stdlib.h>
//Создает объект Model с параметрами по умолчанию. Необходимо вызвать данную функцию для инициализации модели.
Model Model_Create() 
{
	Model n;
	n.Layers = NULL;
	n.n_layers = 0;
	n.NetForward = NULL;
	n.NetBackward = NULL;
	return n;
}
//Добавляет новый слой l в массив Layers модели n и возвращает его адрес.
Layer* Model_AddLayer(Model* n, Layer* l)
{
	int cnt = n->n_layers + 1;
	Layer** tmp = (Layer**)realloc(n->Layers, sizeof(Layer*) * cnt);
	if (!tmp) {
		free(n->Layers);
		n->Layers = NULL;
		return NULL;
	}
	n->n_layers = cnt;
	n->Layers = tmp;
	n->Layers[cnt - 1] = l;
	return n->Layers[cnt - 1];
}
//Общая функция для вызова операции обратного прохода слоя. Вызывает backward функцию соответствующего слоя в зависимости от его типа
void Backward_Layer(Layer* l) 
{
	switch (l->type)
	{
	case LT_INPUT: Input_Backward(l); break;
	case LT_DENSE: Dense_Backward(l); break;
	//case LT_SOFTMAX: break;
	case LT_RELU: Relu_Backward(l); break;
	//case LT_REGRESSION: Regression_Backward(l, y); break;
	//case LT_MSE: MSE_Backward(l,y); break;
	case LT_CONV: Conv2d_Backward(l); break;
	case LT_MAXPOOL: MaxPool2d_Backward(l); break;
	case LT_TANHA: TanhA_Backward(l); break;
	case LT_CONC: Conc_Backward(l); break;
	default:
		break;
	}
}
//Общая функция для вызова операции прямого прохода слоя. Вызывает forward функцию соответствующего слоя в зависимости от его типа.
Tensor *Forward_Layer(Layer* l)
{
	Tensor* y = NULL;
	switch (l->type)
	{
	case LT_INPUT: y = Input_Forward(l); break;
	case LT_DENSE: y = Dense_Forward(l); break;
	case LT_SOFTMAX: break;
	case LT_RELU: y = Relu_Forward(l); break;
	case LT_REGRESSION: y = Regression_Forward(l); break;
	case LT_MSE: y = MSE_Forward(l); break;
	case LT_TANHA: y = TanhA_Forward(l); break;
	case LT_CONV: y = Conv2d_Forward(l); break;
	case LT_MAXPOOL: y = MaxPool2d_Forward(l); break;
	case LT_CONC: y = Conc_Forward(l); break;
	default: break;
	}
	return y;
}
//Загрузка данных из cJSON объекта node в слой t
void Layer_Load_JSON(Layer* t, cJSON* node)
{
	cJSON* output_shape = cJSON_GetObjectItem(node, "os"); //shape
	cJSON* layer_type = cJSON_GetObjectItem(node, "lt"); //type
	cJSON* num_inputs = cJSON_GetObjectItem(node, "ni"); //num_inputs
	cJSON* jData = cJSON_GetObjectItem(node, "d"); //Layer additional data

	if (!cJSON_IsNull(jData))
		//Load layer data
		switch (t->type)
		{
		case LT_DENSE: {
			Dense* data = (Dense*)t->aData;
			Dense_Load_JSON(data, jData);
		}break;
		case LT_CONV: {
			Conv2d* data = (Conv2d*)t->aData;
			Conv2d_Load_JSON(data, jData);
		}break;
		default:
			break;
		}
}
//Конвертирование данных из слоя l в cJSON объект
cJSON* Layer_To_JSON(Layer* l)
{
	cJSON* Layer = cJSON_CreateObject();
	cJSON_AddItemToObject(Layer, "os", Shape_To_JSON(l->out_shape));
	cJSON_AddNumberToObject(Layer, "ni", l->n_inputs);
	cJSON_AddNumberToObject(Layer, "lt", l->type);

	switch (l->type)
	{
	case LT_DENSE: {
		Dense* data = (Dense*)l->aData;
		cJSON_AddItemToObject(Layer, "d", Dense_To_JSON(data));
	}break;
	case LT_CONV: 
	{
		Conv2d* data = (Conv2d*)l->aData;
		cJSON_AddItemToObject(Layer, "d", Conv2d_To_JSON(data));
	}
	default:
		break;
	}
	return Layer;
}
//Конвертирование данных из модели n в cJSON объект
cJSON* Model_To_JSON(Model* n)
{
	cJSON* jNet = cJSON_CreateObject();
	cJSON* jLayers = cJSON_CreateArray();
	cJSON_AddNumberToObject(jNet, "n_layers", n->n_layers);
	for (int i = 0; i < n->n_layers; i++)
	{
		cJSON* jLayer = Layer_To_JSON(n->Layers[i]);
		cJSON_AddItemToArray(jLayers, jLayer);
	}
	cJSON_AddItemToObject(jNet, "Layers", jLayers);
	return jNet;
}
//Загрузка данных из cJSON объекта node в модель t
void Model_Load_JSON(Model* t, cJSON* node)
{
	cJSON* Layers = cJSON_GetObjectItem(node, "Layers"); //Layers
	cJSON* n_layers = cJSON_GetObjectItem(node, "n_layers"); //num_layers
	int n = n_layers->valueint;

	int i = 0;
	cJSON* layer = NULL;
	cJSON_ArrayForEach(layer, Layers)
	{
		Layer_Load_JSON(t->Layers[i], layer);
		i++;
	}
	/*
	for (int i = 0; i < n; i++)
	{
		cJSON* l = cJSON_GetArrayItem(Layers, i);
		Layer_Load_JSON(t->Layers[i], l);
	}
	*/
}
//Функция прямого прохода всех слоев в модели n. 
void Model_Forward(Model* n) 
{
	for (int i = 0; i < n->n_layers; i++)
	{
		Forward_Layer(n->Layers[i]);
	}
}
//Функция обратного прохода всех слоев в модели n. 
void Model_Backward(Model* n) 
{
	int N = n->n_layers;
	for (int i = N - 1; i >= 0; i--)
	{
		Layer* l = n->Layers[i];
		Backward_Layer(l);
	}
}

void Model_CLearGrads(Model* m)
{
	//clear parameters grads
	dList props = Model_getGradients(m);
	for (int i = 0; i < props.length; i++)
	{
		Tensor* target = (Tensor*)props.data[i].e;
		memset(target->dw, 0, sizeof(float) * target->n);
	}
	//clear chain grads
	for (int i = 0; i < m->n_layers; i++)
	{
		Tensor* out = &m->Layers[i]->output;
		memset(out->dw, 0, sizeof(float) * out->n);
	}
	dList_free(&props);
}

//Функция возвращает динамический список из тензоров для обучения
dList Model_getGradients(Model* n)
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