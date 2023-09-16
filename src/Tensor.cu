#include "Tensor.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <curand_kernel.h>

#ifdef __NVCC__
Tensor Tensor_FromDataGPU(shape s, const float* data) 
{
	Tensor t = Tensor_CreateGPU(s, 0.f);
	cudaMemcpy(t.w, data, sizeof(float)*t.n, cudaMemcpyHostToDevice);
	return t;
}

__global__ void Tensor_FillKernel(int limit, float *w, float v) 
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i<limit)
		w[i] = v;
}

void Tensor_FillGPU(Tensor *v, float c) 
{
	int threadsPerBlockX = 32;

	dim3 gridDim(ceil(v->n / (float)threadsPerBlockX), 1, 1);
	dim3 blockDim(threadsPerBlockX, 1, 1);
	Tensor_FillKernel KERNEL_CALL(gridDim, blockDim) (v->n, v->w, c);
	cudaDeviceSynchronize();
}

void Tensor_FillArrayGPU(float* v, int n, float c)
{
	int threadsPerBlockX = 32;
	if (n < threadsPerBlockX) threadsPerBlockX = 1;
	dim3 gridDim(ceil(n / (float)threadsPerBlockX), 1, 1);
	dim3 blockDim(threadsPerBlockX, 1, 1);
	Tensor_FillKernel KERNEL_CALL(gridDim, blockDim) (n, v, c);
	cudaDeviceSynchronize();
}

Tensor Tensor_CreateGPU(shape s, float c)
{
	Tensor v;
	v.s.w = s.w;
	v.s.h = s.h;
	v.s.d = s.d;
	v.n = s.w * s.h * s.d;
	v.sumdw = 0;

	v.w = NULL; 
	v.dw = NULL; 
	//v.vt = NULL;
	v.tData = NULL;

	if (cudaMalloc((void**)&v.w, v.n * sizeof(float)) != cudaSuccess) printf("Tensor weights allocation error\n");
	else Tensor_FillGPU(&v, c);
	if (cudaMalloc((void**)&v.dw, v.n * sizeof(float)) != cudaSuccess) printf("Tensor grads allocation error\n");
	else cudaMemset(v.dw, 0, sizeof(float) * v.n);
	//if (cudaMalloc((void**)&v.vt, v.n * sizeof(float)) != cudaSuccess) printf("Tensor additions allocation error\n");
	//else cudaMemset(v.vt, 0, sizeof(float) * v.n);

	return v;
}
void Tensor_FreeGPU(Tensor* v)
{
	if (cudaFree(v->w) != cudaSuccess) printf("Tensor weights free error\n");
	else v->w = NULL;
	if (cudaFree(v->dw) != cudaSuccess) printf("Tensor grads free error\n");
	else v->dw = NULL;
	//if (cudaFree(v->vt) != cudaSuccess) printf("Tensor additions free error\n");
	//else v->vt = NULL;
}

void Tensor_CopyDataGPU(Tensor* dst, Tensor* src)
{
	cudaMemcpy(dst->w, src->w, sizeof(float) * src->n, cudaMemcpyDeviceToDevice);
}
#endif

//print weights
#ifdef __NVCC__
__global__ void TPrintKernel(float* w, int n)
{
	printf("[");
	for (int i = 0; i < n; i++)
		printf("%f, ", w[i]);
	printf("]\n");
}
void Tensor_PrintGPU(Tensor* v)
{
	TPrintKernel KERNEL_CALL_ONCE(v->w, v->n);
	cudaDeviceSynchronize();
}
void Tensor_PrintArrayGPU(float* v, int n)
{
	TPrintKernel KERNEL_CALL_ONCE(v, n);
	cudaDeviceSynchronize();
}
#endif

//random weights
#ifdef __NVCC__
__global__ void xavier_rand_kernel(void* globalState, float* w, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < n) {
		//calculate the range for the weights
		float lower = -(1.0f / sqrtf(n));
		float upper = (1.0f / sqrtf(n));
		curandState localState = ((curandState*)globalState)[i];
		float num = curand_uniform(&localState);
		//scale to the desired range
		float scaled = lower + num * (upper - lower);
		w[i] = scaled;
		((curandState*)globalState)[i] = localState;
	}
}

__global__ void setup_rng_kernel(int limit, void* state)
{
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < limit) {
		curandState* ls = (curandState*)state;
		curand_init(clock(), id, 0, &ls[id]);
	}
}
void Tensor_Xavier_RandGPU(float *w, int n)
{
	curandState* devStates;
	cudaMalloc(&devStates, n * sizeof(curandState));
	setup_rng_kernel KERNEL_CALL(n, 1) (n, devStates);
	cudaDeviceSynchronize();

	int threadsPerBlockX = 32;

	dim3 gridDim(ceil(n / (float)threadsPerBlockX), 1, 1);
	dim3 blockDim(threadsPerBlockX, 1, 1);
	xavier_rand_kernel KERNEL_CALL(gridDim, blockDim) (devStates, w, n);
	cudaDeviceSynchronize();

	cudaFree(devStates);
}
#endif
//============================================================================================

#ifdef __NVCC__
#endif 
