#ifndef MODEL_H
#define MODEL_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Interfaces.h"
#include "Dense.h"
#include "Input.h"
#include "MSE.h"
#include "TanhA.h"
#include "Conv2d.h"
#include "MaxPool2d.h"
#include "Relu.h"
#include "Regression.h"
#include "Conc.h"
#include "cJSON.h"

typedef struct _Model
{
	Layer** Layers;
	int n_layers;

	Tensor* (*NetForward) (struct _Model* n, Tensor* x);
	void (*NetBackward) (struct _Model* n, Tensor *y);
}Model;

Model Model_Create();
Layer* Model_AddLayer(Model *n, Layer* l);

void Backward_Layer (Layer* l);
Tensor *Forward_Layer(Layer* l);

void Model_Forward(Model* n);
void Model_Backward(Model* n);

cJSON* Layer_To_JSON(Layer* l);
void Layer_Load_JSON(Layer* t, cJSON* node);
cJSON* Model_To_JSON(Model *n);
void Model_Load_JSON(Model *t, cJSON* node);
void Model_CLearGrads(Model* m);
dList Model_getGradients(Model* n);
#ifdef __NVCC__
Model Model_CreateGPU();
Tensor* Forward_LayerGPU(Layer* l);
void Backward_LayerGPU(Layer* l);
void Model_ForwardGPU(Model* n);
void Model_BackwardGPU(Model* n);
Tensor* Seq_ForwardGPU(Model* n, Tensor* x);
void Seq_BackwardGPU(Model* n, Tensor* y);
#endif // __NVCC__


#ifdef __cplusplus
}
#endif

#endif
