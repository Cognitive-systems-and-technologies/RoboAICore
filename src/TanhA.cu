#include "TanhA.h"
#include <stdlib.h>
#include <math.h>

#ifdef __NVCC__
Layer* TanhA_CreateGPU(Layer* in) 
{
	Layer* dl = (Layer *)malloc(sizeof(Layer));
	if (!dl)
	{
		printf("Tanh allocation error!");
		return NULL;
	}
	dl->type = LT_TANHA;
	dl->n_inputs = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	dl->out_shape = { in->out_shape.w, in->out_shape.h, in->out_shape.d };
	dl->output = Tensor_CreateGPU(dl->out_shape, 0);
	dl->input = &in->output;
	dl->aData = NULL;
	return dl;
}

__global__ void TanhA_ForwardKernels(float* xw, float* outw)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	outw[i] = tanhf(xw[i]);
}

Tensor* TanhA_ForwardGPU(Layer* l)
{
	int n = l->n_inputs;

	int threadsPerBlockX = 128;
	if (n < threadsPerBlockX) threadsPerBlockX = 1;
	dim3 gridDim(ceil(n / (float)threadsPerBlockX), 1, 1);
	dim3 blockDim(threadsPerBlockX, 1, 1);

	TanhA_ForwardKernels KERNEL_CALL(gridDim, blockDim) (l->input->w, l->output.w);
	cudaDeviceSynchronize();
	return &l->output;
}

__global__ void TanhA_BackwardKernels(float* xdw, float* outw, float* outdw)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	float xwi = outw[i];
	float dw = (1.f - xwi * xwi) * outdw[i];
	//atomicAdd(&xdw[i], dw);
	xdw[i] += dw;
}

void TanhA_BackwardGPU(Layer* l) 
{
	int n = l->n_inputs;

	int threadsPerBlockX = 128;
	if (n < threadsPerBlockX) threadsPerBlockX = 1;
	dim3 gridDim(ceil(n / (float)threadsPerBlockX), 1, 1);
	dim3 blockDim(threadsPerBlockX, 1, 1);

	TanhA_BackwardKernels KERNEL_CALL(gridDim, blockDim) (l->input->dw, l->output.w, l->output.dw);
	cudaDeviceSynchronize();
}
#endif // __NVCC__