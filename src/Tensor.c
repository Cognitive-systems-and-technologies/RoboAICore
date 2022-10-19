#include "Tensor.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Tensor *Tensor_Create(shape s, float c, int isTrain) 
{
	Tensor* v = NULL;
	#ifdef __NVCC__
		if (cudaMallocManaged((void**)&v, sizeof(Tensor)) != cudaSuccess) printf("Tensor allocation error\n");
	#else
		v = (Tensor*)malloc(sizeof(Tensor));
	#endif
	
	v->s.w = s.w;
	v->s.h = s.h;
	v->s.d = s.d;
	v->n = s.w * s.h * s.d;
	v->gsum = NULL;

	#ifdef __NVCC__
		if (cudaMallocManaged((void**)&v->w, v->n * sizeof(float)) != cudaSuccess) printf("Tensor data allocation error\n");
		if (cudaMallocManaged((void**)&v->dw, v->n * sizeof(float)) != cudaSuccess) printf("Tensor data allocation error\n");
		if (isTrain)
			if (cudaMallocManaged((void**)&v->gsum, v->n * sizeof(float)) != cudaSuccess) printf("Tensor data allocation error\n");
	#else
		v->w = (float*)malloc(v->n * sizeof(float));
		v->dw = (float*)malloc(v->n * sizeof(float));
		if (isTrain)
			v->gsum = (float*)malloc(v->n * sizeof(float));
	#endif

	for (size_t i = 0; i < v->n; i++) {
		//float x = (float)rand() / (float)(RAND_MAX / c);
		v->w[i] = c;
		v->dw[i] = 0;
		if (isTrain)
			v->gsum[i] = 0;
	}
	return v;
}

void Tensor_Init(Tensor *v, shape s, float c, int isTrain)
{
	v->s.w = s.w;
	v->s.h = s.h;
	v->s.d = s.d;
	v->n = s.w * s.h * s.d;

	v->w = (float*)malloc(v->n*sizeof(float));
	v->dw = (float*)malloc(v->n*sizeof(float));
	v->gsum = NULL;
	if (isTrain)
		v->gsum = (float*)malloc(v->n*sizeof(float));

	for (size_t i = 0; i < v->n; i++) {
		//float x = (float)rand() / (float)(RAND_MAX / c);
		v->w[i] = c;
		v->dw[i] = 0;
		if (isTrain)
			v->gsum[i] = 0;
	}
}

cJSON* Tensor_To_JSON(Tensor* v) 
{
	cJSON* root = NULL;
	cJSON* fld = NULL;
	root = cJSON_CreateArray();
	cJSON_AddItemToArray(root, fld = cJSON_CreateObject());
	cJSON_AddNumberToObject(fld, "sw", v->s.w);
	cJSON_AddNumberToObject(fld, "sh", v->s.h);
	cJSON_AddNumberToObject(fld, "sd", v->s.d);

	cJSON_AddNumberToObject(fld, "n", v->n);
	cJSON *w = cJSON_CreateFloatArray(v->w, v->n);
	cJSON_AddItemToObject(fld, "w", w);

	cJSON* dw = cJSON_CreateFloatArray(v->dw, v->n);
	cJSON_AddItemToObject(fld, "dw", dw);

	cJSON* gsum = cJSON_CreateFloatArray(v->gsum, v->n);
	cJSON_AddItemToObject(fld, "gsum", gsum);

	return root;
}

float Tensor_WeightedSum(Tensor* v1, Tensor* v2)
{
	float a = 0.0f;
	if (v1->n != v2->n)
		return a;
	int n = v1->n;
	for (int i = 0; i < n; i++)
	{
		a += v1->w[i] * v2->w[i];
	}
	return a;
}

void Tensor_Free(Tensor *v)
{
	#ifdef __NVCC__
		if (cudaFree(v->w) != cudaSuccess) printf("Tensor data free error\n");
		if (cudaFree(v->dw) != cudaSuccess) printf("Tensor data free error\n");
		if (cudaFree(v->gsum) != cudaSuccess) printf("Tensor data free error\n");
		if (cudaFree(v) != cudaSuccess) printf("Tensor free error\n");
	#else
		free(v->dw);
		free(v->w);
		free(v->gsum);
		free(v);
	#endif 
}

//vol functions
float Tensor_Get(Tensor *vol, int x, int y, int d)
{
	int ix = ((vol->s.w * y) + x) * vol->s.d + d;
	return vol->w[ix];
}

void Tensor_Set(Tensor *vol, int w, int h, int d, float v)
{
	int ix = ((vol->s.w * h) + w) * vol->s.d + d;
	vol->w[ix] = v;
}

void Tensor_Copy(Tensor* dst, Tensor* src) 
{
	memcpy(dst->w, src->w, sizeof(float) * src->n);
	memcpy(dst->dw, src->dw, sizeof(float) * src->n);
	if (src->gsum!=NULL&&dst->gsum!=NULL)
		memcpy(dst->w, src->w, sizeof(float) * src->n);
}


#ifdef __NVCC__
__global__ void Tensor_PrintKernel(Tensor* v)
{
	int i = threadIdx.x;
	printf("%f\n", v->w[i]);
}
void Tensor_Print(Tensor* v)
{
	Tensor_PrintKernel KERNEL_CALL(1, v->n) (v);
}
#endif 
