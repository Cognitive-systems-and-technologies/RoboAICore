#include "Optimizer.h"
#include <stdlib.h>
#include <math.h>

#ifdef __NVCC__
void CreateAdanDataGPU(Tensor* t) 
{
	adanTData* data = (adanTData*)malloc(sizeof(adanTData));;
	data->gprev = createFloatArrayGPU(t->n);
	data->mk = createFloatArrayGPU(t->n);
	data->nk = createFloatArrayGPU(t->n);
	data->vk = createFloatArrayGPU(t->n);

	if (data->mk!=NULL && data->vk != NULL && data->nk != NULL && data->gprev != NULL) {
		t->tData = data;
	}
	else printf("Adan GPU data allocation error!\n");
}

void CreateAdamDataGPU(Tensor* t) 
{
	adamTData* data = (adamTData*)malloc(sizeof(adamTData));;
	data->vt = createFloatArrayGPU(t->n);
	data->mt = createFloatArrayGPU(t->n);

	if (data->vt != NULL && data->mt != NULL) {
		t->tData = data;
	}
	else printf("Adam GPU data allocation error!\n");
}

void CreateMomentumDataGPU(Tensor* t) 
{
	momentumTData* data = (momentumTData*)malloc(sizeof(momentumTData));;
	data->vk = createFloatArrayGPU(t->n);
	if (data->vk!=NULL) {
		t->tData = data;
	}
	else printf("Momentum GPU data allocation error!\n");
}

__global__ void NRMSProp_GradKernel(float* w, float* dw, float* vt, float* bw, float* bdw, float* bvt, float lr, shape s)
{
	int y = (blockIdx.x * blockDim.x) + threadIdx.x;
	int z = (blockIdx.y * blockDim.y) + threadIdx.y;
	int x = (blockIdx.z * blockDim.z) + threadIdx.z;;

	if (y < s.h && z < s.d && x < s.w) {
		int i = ((s.w * y) + x) * s.d + z;

		float b = 0.9f;
		float clip = 1e10f;

		float dwij = dw[i];
		//NRMSProp
		if (dwij > clip)
			dwij = clip;
		if (dwij < -clip)
			dwij = -clip;

		float dx = vt[i];
		vt[i] = vt[i] * b + lr * dwij;
		dx = b * dx + (1.f - b) * vt[i];
		w[i] += -dx;

		dw[i] = 0;

		if (y == 0) //biases
		{
			float dwijb = bdw[i];
			//NRMSProp
			if (dwijb > clip)
				dwijb = clip;
			if (dwijb < -clip)
				dwijb = -clip;

			float dxb = bvt[i];
			bvt[i] = bvt[i] * b + lr * dwijb;
			dxb = b * dxb + (1.f - b) * bvt[i];
			bw[i] += -dxb;

			bdw[i] = 0;
		}
	}
}

void Change_GradGPU(OptParams* par, Tensor* k, Tensor* b, bool norm)
{
	int w = k->s.w;
	int h = k->s.h;
	int d = k->s.d;

	int threadsPerBlockX = 4;
	int threadsPerBlockY = 64;
	int threadsPerBlockZ = 4;
	dim3 gridDim(ceil(h / (float)threadsPerBlockX), ceil(d / (float)threadsPerBlockY), ceil(w / (float)threadsPerBlockZ));
	dim3 blockDim(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);

	//========================================
	switch (par->method)
	{
	//case ADAGRAD: AdagradOpt(v, par); break;
	//case RMSPROP: NRMSPropOpt(v, par);	break;
	case NRMSPROP: 
	{	
		momentumTData* kdata = (momentumTData*)k->tData;
		momentumTData* bdata = (momentumTData*)b->tData;
	
		NRMSProp_GradKernel KERNEL_CALL(gridDim, blockDim) (
			k->w, k->dw, kdata->vk,
			b->w, b->dw, bdata->vk,
			par->learning_rate, k->s);
	}break;
	//case SGD: SGDOpt(v, par); break;
	//case ADAN: AdanOpt(v, par);	break;
	//case ADAM: AdamOpt(v, par);	break;
	default: printf("Currently only NRMSPROP available on GPU\n"); break;
	}
	cudaDeviceSynchronize();
}

void PrepareTensorGPU(Tensor *v, OptParams* par)
{
	if (v->tData == NULL)
		switch (par->method)
		{
		case ADAN: CreateAdanDataGPU(v); break;
		case ADAM: CreateAdamDataGPU(v); break;
		case SGD:break;//no data for simple sgd
		default: CreateMomentumDataGPU(v);	break;
		}
}

void PrepareTDataGPU(Model* n, OptParams* par)
{
	for (int i = 0; i < n->n_layers; i++)
	{
		switch (n->Layers[i]->type)
		{
		case LT_DENSE: {
			Dense* data = (Dense*)n->Layers[i]->aData;
			PrepareTensorGPU(data->kernels, par);
			PrepareTensorGPU(&data->biases, par);
		}break;
		case LT_CONV: {
			printf("Currently Conv2d layer unavailable for optimization on GPU\n");
		}break;
		default: break;
		}
	}
}

void OptimizeModelGPU(Model* n, OptParams* par) 
{
	for (int i = 0; i < n->n_layers; i++)
	{
		switch (n->Layers[i]->type)
		{
		case LT_DENSE: {
			Dense* data = (Dense*)n->Layers[i]->aData;
			Layer* l = n->Layers[i];
			Change_GradGPU(par, data->kernels, &data->biases, false);
			//
			Tensor* out = &l->output;
			cudaMemset(out->dw, 0, sizeof(float) * out->n);
		}break;
		case LT_CONV: {
			Tensor* out = &n->Layers[i]->output;
			cudaMemset(out->dw, 0, sizeof(float) * out->n);
		}break;
		default: {
			Tensor* out = &n->Layers[i]->output;
			cudaMemset(out->dw, 0, sizeof(float) * out->n);
		}
			   break;
		}
	}
}
#endif // __NVCC__