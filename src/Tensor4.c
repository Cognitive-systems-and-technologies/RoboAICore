#include "Tensor4.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Tensor4 Tensor4_Create(shape4 s, float c)
{
	Tensor4 v;
	v.s.w = s.w;
	v.s.h = s.h;
	v.s.d = s.d;
	v.s.b = s.b;

	v.n = s.w * s.h * s.d * s.b;
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

void Tensor4_CopyData(Tensor4* dst, Tensor4* src)
{
	memcpy(dst->w, src->w, sizeof(float) * src->n);
}

void Tensor4_Free(Tensor4* v)
{
	free(v->dw);
	v->dw = NULL;
	free(v->w);
	v->w = NULL;
	free(v->vt);
	v->vt = NULL;
}

//============================================================================================
int tIdx4(shape4 s, int w, int h, int d, int b)
{
	return (((s.w * h) + w) * s.d + d)*s.b+b;
}

void Tensor4_Set(Tensor4* t, int w, int h, int d, int b, float v)
{
	int id = (((t->s.w * h) + w) * t->s.d + d) * t->s.b + b;
	t->w[id] = v;
}
float Tensor4_Get(Tensor4* t, int w, int h, int d, int b)
{
	int id = (((t->s.w * h) + w) * t->s.d + d) * t->s.b + b;
	return t->w[id];
}

void Tensor4_Copy(Tensor4* dst, Tensor4* src) 
{
	memcpy(dst->w, src->w, sizeof(float) * src->n);
	memcpy(dst->dw, src->dw, sizeof(float) * src->n);
	if (src->vt!=NULL&&dst->vt!=NULL)
		memcpy(dst->w, src->w, sizeof(float) * src->n);
}

void T4Print(Tensor4 *t)
{
	for (size_t b = 0; b < t->s.b; b++)
	{
		printf("\n\n");
		for (size_t d = 0; d < t->s.d; d++)
		{
			printf("[\n");
			for (size_t h = 0; h < t->s.h; h++)
			{
				printf("[");
				for (size_t w = 0; w < t->s.w; w++)
				{
					int id = tIdx4(t->s, w, h, d, b);
					float x = t->w[id];
					printf("%f, ", x);
				}
				printf("]\n");
			}
			printf("]\n");
		}
		//printf("]");
	}
	printf("\n");
}