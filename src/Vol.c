#include "Vol.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Vol *Vol_Create(shape s, float c, int isTrain) 
{
	Vol* v = NULL;
	#ifdef __NVCC__
		if (cudaMallocManaged((void**)&v, sizeof(Vol)) != cudaSuccess) printf("Vol allocation error\n");
	#else
		v = (Vol*)malloc(sizeof(Vol));
	#endif
	
	v->s.w = s.w;
	v->s.h = s.h;
	v->s.d = s.d;
	v->n = s.w * s.h * s.d;
	v->gsum = NULL;

	#ifdef __NVCC__
		if (cudaMallocManaged((void**)&v->w, v->n * sizeof(float)) != cudaSuccess) printf("Vol data allocation error\n");
		if (cudaMallocManaged((void**)&v->dw, v->n * sizeof(float)) != cudaSuccess) printf("Vol data allocation error\n");
		if (isTrain)
			if (cudaMallocManaged((void**)&v->gsum, v->n * sizeof(float)) != cudaSuccess) printf("Vol data allocation error\n");
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

void Vol_Init(Vol *v, shape s, float c, int isTrain)
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

cJSON* Vol_To_JSON(Vol* v) 
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

float Vol_WeightedSum(Vol* v1, Vol* v2)
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

void Vol_Free(Vol *v)
{
	#ifdef __NVCC__
		if (cudaFree(v->w) != cudaSuccess) printf("Vol data free error\n");
		if (cudaFree(v->dw) != cudaSuccess) printf("Vol data free error\n");
		if (cudaFree(v->gsum) != cudaSuccess) printf("Vol data free error\n");
		if (cudaFree(v) != cudaSuccess) printf("Vol free error\n");
	#else
		free(v->dw);
		free(v->w);
		free(v->gsum);
		free(v);
	#endif 
}

//vol functions
float Vol_Get(Vol *vol, int x, int y, int d)
{
	int ix = ((vol->s.w * y) + x) * vol->s.d + d;
	return vol->w[ix];
}

void Vol_Set(Vol *vol, int w, int h, int d, float v)
{
	int ix = ((vol->s.w * h) + w) * vol->s.d + d;
	vol->w[ix] = v;
}

void Vol_Copy(Vol* dst, Vol* src) 
{
	memcpy(dst->w, src->w, sizeof(float) * src->n);
	memcpy(dst->dw, src->dw, sizeof(float) * src->n);
	if (src->gsum!=NULL&&dst->gsum!=NULL)
		memcpy(dst->w, src->w, sizeof(float) * src->n);
}


#ifdef __NVCC__
__global__ void Vol_PrintKernel(Vol* v)
{
	int i = threadIdx.x;
	printf("%f\n", v->w[i]);
}
void Vol_Print(Vol* v)
{
	Vol_PrintKernel KERNEL_CALL(1, v->n) (v);
}
#endif 
