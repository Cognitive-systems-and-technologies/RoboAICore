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
	v.vt = NULL;

	v.w = (float *)malloc(sizeof(float) * v.n);
	v.dw = (float*)malloc(sizeof(float) * v.n);
	v.vt = (float*)malloc(sizeof(float) * v.n);

	v.sumdw = 0;
	if (!v.w || !v.dw) printf("Tensor data allocation error");
	else
		for (int i = 0; i < v.n; i++) {
			v.w[i] = c;
			v.dw[i] = 0;
			v.vt[i] = 0;
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
	free(v->vt);
	v->vt = NULL;
}

void Tensor_Xavier_Rand(float *w, int n) 
{
	for (int i = 0; i < n; i++)
	{
		float v = xavier_rand(n);
		w[i] = v;
	}
}
//============================================================================================

float Tensor_Get(Tensor *vol, int x, int y, int d)
{
	int ix = (vol->s.w * y + x) * vol->s.d + d;
	return vol->w[ix];
}

float tIdx(shape s, int w, int h, int d) 
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
	if (src->vt!=NULL&&dst->vt!=NULL)
		memcpy(dst->w, src->w, sizeof(float) * src->n);
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
	for (size_t i = 1; i < t->n; ++i)
	{
		if (t->w[i] < min_v) min_v = t->w[i];
	}
	return min_v;
}

float T_MaxValue(Tensor* t) 
{
	float max_v = t->w[0];
	for (size_t i = 1; i < t->n; ++i)
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
