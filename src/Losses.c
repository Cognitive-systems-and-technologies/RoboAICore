#include "Losses.h"

float MSE_Loss(Tensor* y, Tensor* y_true) 
{
	float sum = 0;
	for (int i = 0; i < y->n; i++)
	{
		float dy = (2.f / (float)y->n) * (y->w[i] - y_true->w[i]);
		y->dw[i] += dy;

		float t = y_true->w[i] - y->w[i];
		sum += t * t;
	}
	float loss = sum / (float)y->n;
	return loss;
}

Tensor SoftmaxProb(Tensor *t) 
{
	Tensor out = Tensor_Create(t->s, 0);
	//get max
	//float maxv = T_MaxValue(t);
	float sum = 0;
	for (size_t i = 0; i < t->n; i++)
	{
		float e = expf(t->w[i]);
		sum += e;
		out.w[i] = e;
	}
	//normalize
	for (size_t i = 0; i < t->n; i++)
	{
		float x = out.w[i] / sum;
		out.w[i] = x;
	}
	return out;
}

float Cross_entropy_Loss(Tensor* y, int idx) 
{
	Tensor x = SoftmaxProb(y);
	for (size_t i = 0; i < y->n; i++)
	{
		float y_true = (i == idx) ? 1.f : 0.f;
		float der = -(y_true - x.w[i]);
		y->dw[i] += der;
	}
	float loss = -logf(x.w[idx]);
	Tensor_Free(&x);
	return loss;
}

float Regression_Loss(Tensor* y, int idx, float val)
{
	float dy = y->w[idx] - val;
	y->dw[idx] += dy;

	float dy2 = dy * dy;
	float loss = 0.5f * dy2;
	return loss;
}