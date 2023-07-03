#include "Dense.h"
#include <stdlib.h>

#ifdef __NVCC__
Layer* Dense_CreateGPU(int num_neurons, Layer* in)
{
	Layer* l = (Layer*)malloc(sizeof(Layer));
	if (!l)
	{
		printf("Dense allocation error!");
		return NULL;
	}
	l->type = LT_DENSE;
	int inn = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	//common layer def
	l->out_shape = { 1, 1, num_neurons };
	l->n_inputs = inn;
	l->output = Tensor_CreateGPU(l->out_shape, 0);
	l->input = &in->output;

	float bias = 0.0f;

	Dense* ld = (Dense*)malloc(sizeof(Dense));
	if (ld) {
		ld->kernels = (Tensor*)malloc(sizeof(Tensor));
		if (ld->kernels) {
			shape kernels_shape = { 1, inn, num_neurons };//each row is weight
			*ld->kernels = Tensor_CreateGPU(kernels_shape, 1.f);
			Tensor_Xavier_RandGPU(ld->kernels->w, ld->kernels->n);
			ld->biases = Tensor_CreateGPU({ 1, 1, num_neurons }, bias);
		}
	}
	else printf("Dense data allocation error\n");
	l->aData = ld;
	printf("Dense, output shape: [%d, %d, %d]\n", l->out_shape.w, l->out_shape.h, l->out_shape.d);
	return l;
}

__global__ void Dense_ForwardKernels(shape limit, float* x, float* k, float* out, shape s)
{
	int h = (blockIdx.x * blockDim.x) + threadIdx.x;
	int d = (blockIdx.y * blockDim.y) + threadIdx.y;
	int w = 0;
	if (h < limit.h && d < limit.d) {

		int id = ((s.w * h) + w) * s.d + d;

		float xi = x[h];
		float wi = k[id];
		float mul = xi * wi;

		atomicAdd(&out[d], mul);
	}
}

Tensor* Dense_ForwardGPU(Layer* l)
{
	Dense* data = (Dense*)l->aData;
	//cudaMemset(l->output.w, 0, sizeof(float) * l->output.n);
	Tensor_CopyDataGPU(&l->output, &data->biases);
	//=====================
	int n = l->input->n;
	int nk = l->out_shape.d;

	int threadsPerBlockX = 16;
	int threadsPerBlockY = 64;

	dim3 gridDim((int)ceil(n / (float)threadsPerBlockX), (int)ceil(nk / (float)threadsPerBlockY), 1);
	dim3 blockDim(threadsPerBlockX, threadsPerBlockY, 1);

	Dense_ForwardKernels KERNEL_CALL(gridDim, blockDim) ({1,n,nk},
		l->input->w, data->kernels->w, l->output.w, data->kernels->s);
	cudaDeviceSynchronize();
	return &l->output;
}

__global__ void Dense_BackwardKernels(shape limit, float* xw, float* xdw, float* kw, float* kdw, float* bdw, float* outdw, shape s)
{
	int h = (blockIdx.x * blockDim.x) + threadIdx.x;
	int d = (blockIdx.y * blockDim.y) + threadIdx.y;
	int w = 0;
	if (h < limit.h && d < limit.d) {
		int id = ((s.w * h) + w) * s.d + d;

		float chain_grad = outdw[d];

		float xgrad = kw[id] * chain_grad;
		float kgrad = xw[h] * chain_grad;

		//kdw[id] += kgrad;
		atomicAdd(&xdw[h], xgrad);
		atomicAdd(&kdw[id], kgrad);

		if (h == 0)
		{
			atomicAdd(&bdw[d], chain_grad);
			//bdw[d] += chain_grad;
		}
	}
}

void Dense_BackwardGPU(Layer* l)
{
	Dense* data = (Dense*)l->aData;
	//=====================
	int n = l->input->n;
	int nk = l->out_shape.d;

	int threadsPerBlockX = 16;
	int threadsPerBlockY = 64;

	dim3 gridDim((int)ceil(n / (float)threadsPerBlockX), (int)ceil(nk / (float)threadsPerBlockY), 1);
	dim3 blockDim(threadsPerBlockX, threadsPerBlockY, 1);

	Dense_BackwardKernels KERNEL_CALL(gridDim, blockDim) ({1, n, nk},
		l->input->w, l->input->dw, data->kernels->w, data->kernels->dw, data->biases.dw, l->output.dw, data->kernels->s);
	cudaDeviceSynchronize();
}
#endif // __NVCC__