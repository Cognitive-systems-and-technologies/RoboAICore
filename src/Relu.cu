#include "Relu.h"
#include <stdlib.h>

#ifdef __NVCC__
Layer* Relu_CreateGPU(Layer* in) 
{
	Layer* dl = (Layer*)malloc(sizeof(Layer));
	if (!dl)
	{
		printf("Relu allocation error!");
		return NULL;
	}
	dl->type = LT_RELU;
	dl->aData = NULL;
	dl->n_inputs = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	dl->out_shape = { in->out_shape.w, in->out_shape.h, in->out_shape.d };
	dl->output = Tensor_CreateGPU(dl->out_shape, 0);
	dl->input = &in->output;
	printf("Relu_GPU, output shape: [%d, %d, %d]\n", dl->out_shape.w, dl->out_shape.h, dl->out_shape.d);
	return dl;
}

__global__ void Relu_ForwardKernels(int limit, float* xw, float* outw)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < limit) {
		//outw[i] = (xw[i] < 0) ? 0 : xw[i];
		if (xw[i] < 0) outw[i] = 0.0001f* xw[i];
		else
			outw[i] = xw[i];
	}
}

Tensor* Relu_ForwardGPU(Layer* l) 
{
	int n = l->n_inputs;

	int threadsPerBlockX = 256;

	dim3 gridDim((int)ceil(n / (float)threadsPerBlockX), 1, 1);
	dim3 blockDim(threadsPerBlockX, 1, 1);

	Relu_ForwardKernels KERNEL_CALL(gridDim, blockDim) (n,
		l->input->w, l->output.w);
	cudaDeviceSynchronize();
	return &l->output;
}

__global__ void Relu_BackwardKernels(int limit, float* xdw, float* outw, float* outdw)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < limit) {
		//xdw[i] += (outw[i] <= 0) ? 0 : outdw[i];
		//if (outw[i] <= 0) xdw[i] += 0;
		//else
			//atomicAdd(&xdw[i], outdw[i]);
			//xdw[i] += outdw[i];
		if (outw[i] < 0) atomicAdd(&xdw[i], 0.0001f* outdw[i]);
		else
			atomicAdd(&xdw[i], outdw[i]);
	}
}

void Relu_BackwardGPU(Layer* l) 
{
	int n = l->n_inputs;

	int threadsPerBlockX = 256;

	dim3 gridDim((int)ceil(n / (float)threadsPerBlockX), 1, 1);
	dim3 blockDim(threadsPerBlockX, 1, 1);

	Relu_BackwardKernels KERNEL_CALL(gridDim, blockDim) (n,
		l->input->dw, l->output.w, l->output.dw);
	cudaDeviceSynchronize();
}
#endif // __NVCC__
