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
		v = malloc(sizeof(Tensor));
	#endif
	if (!v)
	{
		printf("Tensor allocation error!");
		return NULL;
	}
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
		v->w = malloc(sizeof(float) * v->n);
		v->dw = malloc(sizeof(float) * v->n);
		if (isTrain)
			v->gsum = malloc(sizeof(float) * v->n);
	#endif
	if (!v->w)
	{
		printf("w allocation error!");
		return NULL;
	}
	if (!v->dw)
	{
		printf("dw allocation error!");
		return NULL;
	}
	if (isTrain && !v->gsum)
	{
		printf("gsum allocation error!");
		return NULL;
	}

	for (int i = 0; i < v->n; i++) {
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

	v->w = malloc(sizeof(float) * v->n);
	v->dw = malloc(sizeof(float) * v->n);
	v->gsum = NULL;
	if (isTrain)
		v->gsum = malloc(sizeof(float)*v->n);
	if (!v->w)
	{
		printf("w allocation error!");
		v = NULL;
	}
	if (!v->dw)
	{
		printf("dw allocation error!");
		v = NULL;
	}
	if (isTrain && !v->gsum)
	{
		printf("gsum allocation error!");
		v = NULL;
	}
	for (int i = 0; i < v->n; i++) {
		//float x = (float)rand() / (float)(RAND_MAX / c);
		v->w[i] = c;
		v->dw[i] = 0;
		if (isTrain)
			v->gsum[i] = 0;
	}
}

cJSON* Shape_To_JSON(shape s) 
{
	/*cJSON* fld = cJSON_CreateObject();
	const int sh[3] = {s.w, s.h, s.d};
	cJSON* arr = cJSON_CreateIntArray(sh, 3);
	cJSON_AddItemToObject(fld, "shape", arr);
	*/
	const int sh[3] = { s.w, s.h, s.d };
	cJSON* arr = cJSON_CreateIntArray(sh, 3);
	return arr;
}

cJSON* Tensor_To_JSON(Tensor* v) 
{
	//cJSON* root = cJSON_CreateObject();
	cJSON* fld = cJSON_CreateObject();

	//cJSON* shp = Shape_To_JSON(v->s);
	//cJSON_AddItemReferenceToObject(fld,"awdawdawd", shp);

	
	cJSON_AddItemToObject(fld, "s", Shape_To_JSON(v->s));

	cJSON *w = cJSON_CreateFloatArray(v->w, v->n);
	cJSON_AddItemToObject(fld, "w", w);

	cJSON* dw = cJSON_CreateFloatArray(v->dw, v->n);
	cJSON_AddItemToObject(fld, "dw", dw);

	cJSON* gsum = cJSON_CreateFloatArray(v->gsum, v->n);
	cJSON_AddItemToObject(fld, "gs", gsum);

	return fld;
}

Tensor* Tensor_From_JSON(cJSON *node) 
{
	cJSON* shp = cJSON_GetObjectItem(node, "s");
	cJSON* w = cJSON_GetObjectItem(node, "w");
	cJSON* dw = cJSON_GetObjectItem(node, "dw");
	
	shape ts = (shape){ cJSON_GetArrayItem(shp, 0)->valueint,cJSON_GetArrayItem(shp, 1)->valueint,cJSON_GetArrayItem(shp, 2)->valueint };
	Tensor* t = Tensor_Create(ts, 0, 0);
	int n = ts.w * ts.h * ts.d;
	for (int i = 0; i < n; i++)
	{
		t->w[i] = (float)cJSON_GetArrayItem(w, i)->valuedouble;
		t->dw[i] = (float)cJSON_GetArrayItem(dw, i)->valuedouble;
	}
	return t;
}

void Tensor_Load_JSON(Tensor* t, cJSON* node)
{
	cJSON* shp = cJSON_GetObjectItem(node, "s");
	cJSON* w = cJSON_GetObjectItem(node, "w");
	cJSON* dw = cJSON_GetObjectItem(node, "dw");

	shape ts = (shape){ cJSON_GetArrayItem(shp, 0)->valueint,cJSON_GetArrayItem(shp, 1)->valueint,cJSON_GetArrayItem(shp, 2)->valueint };

	int n = ts.w * ts.h * ts.d;
	for (int i = 0; i < n; i++)
	{
		t->w[i] = (float)cJSON_GetArrayItem(w, i)->valuedouble;
		t->dw[i] = (float)cJSON_GetArrayItem(dw, i)->valuedouble;
	}
}

float Tensor_WeightedSum(Tensor* v1, Tensor* v2)
{
	float a = 0.0f;
	if (v1->n != v2->n)
		return a;
	int n = v1->n;
	for (int i = 0; i < n; i++)
	{
		if ((v1->w[i] < 0.0f) == (v2->w[i] < 0.0f)
			&& fabs(v2->w[i]) > FLT_MAX - fabs(v1->w[i])) {
			printf("WEIGHTED_SUM_OVERFLOW");
			return 0;
		}
		else {
			float mul = v1->w[i] * v2->w[i];
			a += mul;
		}
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
		v->dw = NULL;
		free(v->w);
		v->w = NULL;
		free(v->gsum);
		v->gsum = NULL;
		free(v);
		v = NULL;
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
	dst->n = src->n;
	dst->s = src->s;
}

shape T_Argmax(Tensor* t) 
{
	shape idx = {0,0,0};
	float max = t->w[0];
	for (int w = 0; w < t->s.w; w++)
	{
		for (int h = 0; h < t->s.h; h++)
		{
			for (int d = 0; d < t->s.d; d++)
			{
				float val = Tensor_Get(t, w, h, d);
				if (val > max) 
				{
					max = val;
					idx = (shape){w, h, d};
				}
			}
		}
	}
	return idx;
}

Tensor* Tensor_CreateCopy(Tensor* v) 
{
	Tensor* t = Tensor_Create(v->s, 0, 0);
	if (!t) 
	{
		printf("Tensor from copy error");
		return NULL;
	}
	Tensor_Copy(t, v);
	return t;
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
