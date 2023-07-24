#include "MSE.h"

#ifdef __NVCC__
Layer* MSE_CreateGPU(Layer* in)
{
	Layer* l = (Layer*)malloc(sizeof(Layer));
	if (!l)
	{
		printf("MSE allocation error!");
		return NULL;
	}
	l->type = LT_MSE;
	l->n_inputs = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	l->out_shape = { 1, 1, l->n_inputs };
	l->output = Tensor_CreateGPU(l->out_shape, 0);
	l->input = &in->output;

	LData* ld = (LData*)malloc(sizeof(LData));
	if (ld) {
		ld->loss = 0;
	}
	else printf("MSE data allocation error\n");
	l->aData = ld;
	printf("Mse, output shape: [%d, %d, %d]\n", l->out_shape.w, l->out_shape.h, l->out_shape.d);
	return l;
}

Tensor* MSE_ForwardGPU(Layer* l) 
{
	Tensor_CopyDataGPU(&l->output, l->input);
	return &l->output;
}

__global__ void MSE_BackwardKernels(int limit, float* xw, float* xdw, float* yw, float n, float* sum)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < limit) {
		float dy = (2.f / n) * (xw[i] - yw[i]);
		atomicAdd(&xdw[i], dy);
		//xdw[i] += dy;

		float t = yw[i] - xw[i];
		float t2 = t * t;

		atomicAdd(sum, t2);
	}
}

void MSE_BackwardGPU(Layer* l, Tensor* y_true) 
{
	int n = l->n_inputs;

	int threadsPerBlockX = 128;
	dim3 gridDim((int)ceil(n / (float)threadsPerBlockX), 1, 1);
	dim3 blockDim(threadsPerBlockX, 1, 1);

	float *sumd, *sumh;
	sumh = (float*)malloc(sizeof(float));
	if (cudaMalloc((void**)&sumd, sizeof(float)) != cudaSuccess) printf("in loss allocation\n");
	cudaMemset(sumd, 0, sizeof(float));
	MSE_BackwardKernels KERNEL_CALL(gridDim, blockDim) (n,
		l->input->w, l->input->dw, y_true->w, (float)n, sumd);
	cudaDeviceSynchronize();

	cudaMemcpy(sumh, sumd, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(sumd);
	LData* ld = (LData*)l->aData;
	ld->loss = sumh[0]/(float)n;
	free(sumh);
	cudaDeviceSynchronize();
}
#endif // __NVCC__