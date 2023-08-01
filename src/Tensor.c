#include "Tensor.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Tensor Tensor_Create(shape s, float c)
{
	Tensor v;
	v.s.w = s.w;
	v.s.h = s.h;
	v.s.d = s.d;
	v.n = s.w * s.h * s.d;
	//v.vt = NULL;

	v.tData = NULL;

	v.w = (float *)malloc(sizeof(float) * v.n);
	v.dw = (float*)malloc(sizeof(float) * v.n);
	//v.vt = (float*)malloc(sizeof(float) * v.n);

	v.sumdw = 0;
	if (!v.w || !v.dw) printf("Tensor data allocation error");
	else
		for (int i = 0; i < v.n; i++) {
			v.w[i] = c;
			v.dw[i] = 0;
			//v.vt[i] = 0;
		}
	return v;
}

Tensor Tensor_FromData(shape s, const float* data) 
{
	Tensor t = Tensor_Create(s, 0.f);
	memcpy(t.w, data, sizeof(float) * t.n);
	return t;
}

Tensor Tensor_CreateCopy(Tensor* t)
{
	Tensor v;
	v.s.w = t->s.w;
	v.s.h = t->s.h;
	v.s.d = t->s.d;
	v.n = t->n;
	v.tData = NULL;
	v.w = (float*)malloc(sizeof(float) * v.n);
	v.dw = (float*)malloc(sizeof(float) * v.n);
	v.sumdw = 0;
	if (!v.w || !v.dw) printf("Tensor data allocation error");
	else{
		memcpy(v.w, t->w, sizeof(float) * v.n);
		memset(v.dw, 0, sizeof(float) * v.n);
	}
	return v;
}

Tensor* Tensor_CreateDyn(shape s, float c) 
{
	Tensor* v = (Tensor*)malloc(sizeof(Tensor));
	if (!v) 
	{
		printf("Tensor dynamic allocation error\n");
		return NULL;
	}
	v->s.w = s.w;
	v->s.h = s.h;
	v->s.d = s.d;
	v->n = s.w * s.h * s.d;
	v->tData = NULL;

	v->w = (float*)malloc(sizeof(float) * v->n);
	v->dw = (float*)malloc(sizeof(float) * v->n);
	
	v->sumdw = 0;
	if (!v->w || !v->dw) printf("Tensor data allocation error");
	else
		for (int i = 0; i < v->n; i++) {
			v->w[i] = c;
			v->dw[i] = 0;
		}
	return v;
}

void Tensor_CopyData(Tensor* dst, Tensor* src)
{
	memcpy(dst->w, src->w, sizeof(float) * src->n);
}

void Tensor_Free(Tensor* v)
{
	free(v->dw);
	v->dw = NULL;
	free(v->w);
	v->w = NULL;
}

void Tensor_Xavier_Rand(float *w, int n) 
{
	for (int i = 0; i < n; i++)
	{
		float v = xavier_rand(n);
		w[i] = v;
	}
}
void Tensor_He_Rand(float* w, int n)
{
	for (int i = 0; i < n; i++)
	{
		float v = he_rand(n);
		w[i] = v;
	}
}
//============================================================================================

float Tensor_Get(Tensor *vol, int x, int y, int d)
{
	int ix = (vol->s.w * y + x) * vol->s.d + d;
	return vol->w[ix];
}

int tIdx(shape s, int w, int h, int d) 
{
	return ((s.w * h) + w) * s.d + d;
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
	//if (src->vt!=NULL&&dst->vt!=NULL)
		//memcpy(dst->w, src->w, sizeof(float) * src->n);
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

float T_MinValue(Tensor* t) 
{
	float min_v = t->w[0];
	for (size_t i = 1; i < t->n; i++)
	{
		if (t->w[i] < min_v) min_v = t->w[i];
	}
	return min_v;
}

float T_MaxValue(Tensor* t) 
{
	float max_v = t->w[0];
	for (size_t i = 1; i < t->n; i++)
	{
		if (t->w[i] > max_v) max_v = t->w[i];
	}
	return max_v;
}

float T_Mean(Tensor* t) 
{
	float sum = 0;
	for (size_t i = 0; i < t->n; i++)
	{
		sum += t->w[i];
	}
	return sum / t->n;
}

cJSON* Shape_To_JSON(shape s)
{
	const int sh[3] = { s.w, s.h, s.d };
	cJSON* arr = cJSON_CreateIntArray(sh, 3);
	return arr;
}

cJSON* Tensor_To_JSON(Tensor* v)
{
	cJSON* fld = cJSON_CreateObject();
	cJSON_AddItemToObject(fld, "s", Shape_To_JSON(v->s));
	cJSON* w = cJSON_CreateFloatArray(v->w, v->n);
	cJSON_AddItemToObject(fld, "w", w);
	return fld;
}

void Tensor_Load_JSON(Tensor* t, cJSON* node)
{
	cJSON* shp = cJSON_GetObjectItem(node, "s");
	cJSON* w = cJSON_GetObjectItem(node, "w");
	shape ts = (shape){ cJSON_GetArrayItem(shp, 0)->valueint,cJSON_GetArrayItem(shp, 1)->valueint,cJSON_GetArrayItem(shp, 2)->valueint };
	int n = ts.w * ts.h * ts.d;
	int i = 0;
	cJSON* iterator = NULL;
	cJSON_ArrayForEach(iterator, w)
	{
		t->w[i] = iterator->valuedouble;
		i++;
	}
}

void Tensor_Print(Tensor* x)
{
	for (size_t d = 0; d < x->s.d; d++)
	{
		printf("[\n");
		for (size_t i = 0; i < x->s.w; i++)
		{
			for (size_t j = 0; j < x->s.h; j++)
			{
				printf("%f ", Tensor_Get(x, i, j, d));
			}
			printf("\n");
		}
		printf("]\n");
	}
}
