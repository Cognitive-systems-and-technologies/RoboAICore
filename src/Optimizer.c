#include "Optimizer.h"
#include <stdlib.h>
#include <math.h>

OptParams OptParams_Create() 
{
	//Create default
	OptParams par;
	par.eps = 1e-10f;
	par.learning_rate = 0.0001f;
	par.method = ADAGRAD;
	par.counter = 0;

	par.b = 0.9f;
	par.clip = 1e10f;
	par.b1 = 0.02f;
	par.b2 = 0.08f;
	par.b3 = 0.01f;
	return par;
}

void CreateMomentumData(Tensor* t)
{
	momentumTData* data = (momentumTData*)malloc(sizeof(momentumTData));;
	data->vk = NULL;

	data->vk = (float*)malloc(sizeof(float) * t->n);

	if (data->vk) {
		memset(data->vk, 0, sizeof(float) * t->n);
		t->tData = data;
	}
	else printf("Momentum data allocation error!\n");
}

void CreateAdamData(Tensor* t)
{
	adamTData* data = (adamTData*)malloc(sizeof(adamTData));;
	data->vt = NULL; data->mt = NULL;

	data->vt = (float*)malloc(sizeof(float) * t->n);
	data->mt = (float*)malloc(sizeof(float) * t->n);

	if (data->vt && data->mt) {
		memset(data->vt, 0, sizeof(float) * t->n);
		memset(data->mt, 0, sizeof(float) * t->n);
		t->tData = data;
	}
	else printf("Adam data allocation error!\n");
}

void CreateAdanData(Tensor* t)
{
	adanTData *data = (adanTData*)malloc(sizeof(adanTData));;
	data->gprev = NULL; data->mk = NULL; data->nk = NULL; data->vk = NULL;

	data->nk = (float*)malloc(sizeof(float) * t->n);
	data->mk = (float*)malloc(sizeof(float) * t->n);
	data->vk = (float*)malloc(sizeof(float) * t->n);
	data->gprev = (float*)malloc(sizeof(float) * t->n);

	if (data->mk && data->vk && data->nk && data->gprev) {
		memset(data->nk, 0, sizeof(float) * t->n);
		memset(data->mk, 0, sizeof(float) * t->n);
		memset(data->vk, 0, sizeof(float) * t->n);
		memset(data->gprev, 0, sizeof(float) * t->n);
		t->tData = data;
	}
	else printf("Adan data allocation error!\n");
}

void AdanOpt(Tensor *v, OptParams* par) 
{
	adanTData* data = (adanTData*)v->tData;
	if (data != NULL)
	for (int i = 0; i < v->n; i++)
	{
		float g = v->dw[i];
		float gprev = data->gprev[i];
		float e = par->eps;
		float decay = 0.02f;
		float b1 = 0.02f, b2 = 0.08f, b3 = 0.01f;
		data->mk[i] = (1.f - b1) * data->mk[i] + b1 * g;
		data->vk[i] = (1.f - b2) * data->vk[i] + b2 * (g - gprev);
		float sq = g + (1.f - b2) * (g - gprev);
		data->nk[i] = (1.f - b3) * data->nk[i] + b3 * (sq * sq);

		float nk = par->learning_rate / (sqrtf(data->nk[i]) + e);
		v->w[i] = powf((1.f + decay * par->learning_rate), -1.f) * (v->w[i] - nk * (data->mk[i] + (1.f - b2) * data->vk[i]));

		data->gprev[i] = g;
		v->dw[i] = 0;
	}
	else printf("No adan data for training\n");
}

void AdamOpt(Tensor* v, OptParams* par)
{
	adamTData* data = (adamTData*)v->tData;
	if (data != NULL) {
		float* vt = data->vt;
		float* mt = data->mt;
		float b1 = 0.9f, b2 = 0.999f;
		for (int i = 0; i < v->n; i++)
		{
			float g = v->dw[i];
			mt[i] = b1 * mt[i] + (1.f - b1) * g;
			vt[i] = b2 * vt[i] + (1.f - b2) * (g * g);
			float mtt = mt[i] / (1.f - powf(b1, (float)par->counter));
			float vtt = vt[i] / (1.f - powf(b2, (float)par->counter));
			float neww = -par->learning_rate * (mtt / (sqrtf(vtt) + par->eps));
			v->w[i] += neww;
			v->dw[i] = 0;
		}
	}
	else printf("No adam data for training\n");
}

void AdagradOpt(Tensor* v, OptParams* par)
{
	momentumTData* data = (momentumTData*)v->tData;
	if (data != NULL) {
		float* vk = data->vk;
		for (int i = 0; i < v->n; i++)
		{
			float dw = v->dw[i];
			vk[i] += dw * dw;
			v->w[i] += -(par->learning_rate / sqrtf(vk[i] + par->eps)) * dw;
			v->dw[i] = 0;
		}
	}
	else printf("No adagrad data for training\n");
}

void RMSPropOpt(Tensor* v, OptParams* par)
{
	momentumTData* data = (momentumTData*)v->tData;
	if (data != NULL) {
		float* vk = data->vk;
		for (int i = 0; i < v->n; i++)
		{
			float dw = v->dw[i];
			// gradient clip
			if (dw > par->clip) dw = par->clip;
			if (dw < -par->clip) dw = -par->clip;

			float gs_prev = vk[i];
			float gs2 = (dw * dw);
			float dx = par->b * gs_prev + (1.f - par->b) * gs2;
			vk[i] = dx;

			v->w[i] += -(par->learning_rate / sqrtf(dx + par->eps)) * dw;
			v->dw[i] = 0;
		}
	}
	else printf("No rmsprop data for training\n");
}

void NRMSPropOpt(Tensor* v, OptParams* par)
{
	momentumTData* data = (momentumTData*)v->tData;
	if (data != NULL) {
		float* vk = data->vk;
		for (int i = 0; i < v->n; i++)
		{
			float dw = v->dw[i];
			// gradient clip
			if (dw > par->clip) dw = par->clip;
			if (dw < -par->clip) dw = -par->clip;

			float dx = vk[i];
			vk[i] = dx * par->b + par->learning_rate * dw;
			dx = par->b * dx + (1.f - par->b) * vk[i];
			v->w[i] += -dx;

			v->dw[i] = 0;
		}
	}
	else printf("No nrmsprop data for training\n");
}

void SGDOpt(Tensor* v, OptParams* par)
{
	for (int i = 0; i < v->n; i++)
	{
		float dw = v->dw[i];
		v->w[i] += -par->learning_rate * dw;
		v->dw[i] = 0;
	}
}

void Change_Grad(OptParams* par, Tensor* v, bool norm) 
{
	if (v->tData == NULL)
		switch (par->method)
		{
			case ADAN: CreateAdanData(v); break;
			case ADAM: CreateAdamData(v); break;
			case SGD:break;//no data for simple sgd
			default: CreateMomentumData(v);	break;
		}

	//weights normalization
	/*
	float dwnorm = 0;
	if (norm) {
		for (int i = 0; i < v->n; i++)
		{
			dwnorm += dw[i] * dw[i];
		}
		dwnorm = sqrtf(dwnorm) + 1e-10f;
	}
	*/

	switch (par->method)
	{
		case ADAGRAD: AdagradOpt(v, par); break;
		case RMSPROP: NRMSPropOpt(v, par);	break;
		case NRMSPROP: NRMSPropOpt(v, par);	break;
		case SGD: SGDOpt(v, par); break;
		case ADAN: AdanOpt(v, par);	break;
		case ADAM: AdamOpt(v, par);	break;
		default: break;
	}
}

void Optimize(Model* n, OptParams* par, Tensor* x, Tensor* y)
{
	n->NetForward(n, x);
	n->NetBackward(n, y);

	par->counter++;
	//apply gradients
	for (int i = 0; i < n->n_layers; i++)
	{
		switch (n->Layers[i]->type)
		{
			case LT_DENSE: {
				Dense* data = (Dense*)n->Layers[i]->aData;
				for (size_t i = 0; i < data->biases.n; i++)
				{
					Change_Grad(par, &data->kernels[i], false);
				}
				Change_Grad(par, &data->biases, false);

				Tensor* out = &n->Layers[i]->output;
				memset(out->dw, 0, sizeof(float) * out->n);
			}break;
			case LT_CONV: {
				Conv2d* data = (Conv2d*)n->Layers[i]->aData;
				for (size_t i = 0; i < data->biases.n; i++)
				{
					Change_Grad(par, &data->kernels[i], false);
				}
				Change_Grad(par, &data->biases, false);

				Tensor* out = &n->Layers[i]->output;
				memset(out->dw, 0, sizeof(float) * out->n);
			}break;
			default: {
				Tensor* out = &n->Layers[i]->output;
				memset(out->dw, 0, sizeof(float) * out->n);
			}	
			break;
		}
	}
}

void OptimizeModel(Model* n, OptParams* par)
{
	par->counter++;
	//apply gradients
	for (int i = 0; i < n->n_layers; i++)
	{
		switch (n->Layers[i]->type)
		{
		case LT_DENSE: {
			Dense* data = (Dense*)n->Layers[i]->aData;
			for (size_t i = 0; i < data->biases.n; i++)
			{
				Change_Grad(par, &data->kernels[i], false);
			}
			Change_Grad(par, &data->biases, false);

			Tensor* out = &n->Layers[i]->output;
			memset(out->dw, 0, sizeof(float) * out->n);
		}break;
		case LT_CONV: {
			Conv2d* data = (Conv2d*)n->Layers[i]->aData;
			for (size_t i = 0; i < data->biases.n; i++)
			{
				Change_Grad(par, &data->kernels[i], false);
			}
			Change_Grad(par, &data->biases, false);

			Tensor* out = &n->Layers[i]->output;
			memset(out->dw, 0, sizeof(float) * out->n);
		}break;
		default: {
			Tensor* out = &n->Layers[i]->output;
			memset(out->dw, 0, sizeof(float) * out->n);
		}
			   break;
		}
	}
}