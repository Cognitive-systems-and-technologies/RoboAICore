#include "Losses.h"

#ifdef __NVCC__
__global__ void SoftmaxProbKernels(int n, float* iw, float* ow)
{
	__shared__ float sum[1];
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < n) {
		float e = expf(iw[i]);
		ow[i] = e;
		atomicAdd(&sum[0], e);
		__syncthreads();
		float x = ow[i] / sum[0];
		ow[i] = x;
	}
}
Tensor SoftmaxProbGPU(Tensor* t)
{
	Tensor out = Tensor_CreateGPU(t->s, 0);
	int n = t->n;
	int threadsPerBlockX = 128;
	dim3 gridDim((int)ceil(n / (float)threadsPerBlockX), 1, 1);
	dim3 blockDim(threadsPerBlockX, 1, 1);
	SoftmaxProbKernels KERNEL_CALL(gridDim, blockDim) (n,
		t->w, out.w);
	cudaDeviceSynchronize();
	return out;
}

__global__ void Cross_entropy_LossKernels(int n, float* xw, float* ydw, int idx)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < n) {
		float y_true = (i == idx) ? 1.f : 0.f;
		float der = -(y_true - xw[i]);
		atomicAdd(&ydw[i], der);
	}
}

float Cross_entropy_LossGPU(Tensor* y, int idx)
{
	Tensor x = SoftmaxProbGPU(y);

	int n = y->n;
	int threadsPerBlockX = 128;
	dim3 gridDim((int)ceil(n / (float)threadsPerBlockX), 1, 1);
	dim3 blockDim(threadsPerBlockX, 1, 1);
	Cross_entropy_LossKernels KERNEL_CALL(gridDim, blockDim) (n, x.w, y->dw, idx);
	cudaDeviceSynchronize();
	float true_val = 0.f;
	cudaMemcpy(&true_val, &x.w[idx], sizeof(float), cudaMemcpyDeviceToHost);
	float loss = -logf(true_val);
	Tensor_FreeGPU(&x);
	return loss;
}

__global__ void MSE_LossKernels(int n, float* yw, float* ytw, float* ydw, float* sum)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < n) {
		float dy = (2.f / (float)n) * (yw[i] - ytw[i]);
		atomicAdd(&ydw[i], dy);
		float t = ytw[i] - yw[i];
		float t2 = t * t;
		atomicAdd(sum, t2);
	}
}

float MSE_LossGPU(Tensor* y, Tensor* y_true) 
{
	int n = y->n;
	int threadsPerBlockX = 128;
	dim3 gridDim((int)ceil(n / (float)threadsPerBlockX), 1, 1);
	dim3 blockDim(threadsPerBlockX, 1, 1);
	float sum = 0;
	float* sumd;
	if (cudaMalloc((void**)&sumd, sizeof(float)) != cudaSuccess) printf("in loss allocation\n");
	cudaMemset(sumd, 0, sizeof(float));
	MSE_LossKernels KERNEL_CALL(gridDim, blockDim) (n,
		y->w, y_true->w, y->dw, sumd);
	cudaDeviceSynchronize();
	cudaMemcpy(&sum, sumd, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(sumd);
	float loss = sum / (float)n;
	return loss;
}
#endif // __NVCC__