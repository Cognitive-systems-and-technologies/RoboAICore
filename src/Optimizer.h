#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Model.h"
#include <string.h> 

typedef enum OptMethod {
	ADAGRAD,
	RMSPROP,
	ADAM,
	ADAN,
	NRMSPROP,
	SGD
} OptMethod;

typedef struct OptParams
{
	float learning_rate;
	OptMethod method;
	float eps;
	int counter;
	float b1, b2, b3;
	float decay;
	float b;
	float clip;
}OptParams;

typedef struct adanTData 
{
	float* mk;
	float* vk;
	float* nk;
	float* gprev;
}adanTData;

typedef struct adamTData
{
	float* mt;
	float* vt;
}adamTData;

typedef struct momentumTData
{
	float* vk;
}momentumTData;

void CreateAdanData(Tensor* t);
void CreateAdamData(Tensor* t);
void CreateMomentumData(Tensor* t);

void AdanOpt(Tensor* v, OptParams* par);
void AdamOpt(Tensor* v, OptParams* par);
void AdagradOpt(Tensor* v, OptParams* par);
void RMSPropOpt(Tensor* v, OptParams* par);
void NRMSPropOpt(Tensor* v, OptParams* par);
void SGDOpt(Tensor* v, OptParams* par);

OptParams OptParams_Create();
void Optimize(Model*n, OptParams *par, Tensor *x, Tensor *y);
void Change_Grad(OptParams* par, Tensor* v, bool norm);

#ifdef __NVCC__
__global__ void NRMSProp_GradKernel(shape limit, float* w, float* dw, float* vt, float* bw, float* bdw, float* bvt, float lr, shape s);
__global__ void Change_GradKernel(float* w, float* dw, float* vt, float lr, shape s);
void OptimizeGPU(Model* n, OptParams* par, Tensor* x, Tensor* y);
__global__ void Adagrad_GradKernelLinear(int limit, float* w, float* dw, float* vt, float* bw, float* bdw, float* bvt, int bn, float lr);
__global__ void RMSProp_GradKernelLinear(int limit, float* w, float* dw, float* vt, float* bw, float* bdw, float* bvt, int bn, float lr);
__global__ void NRMSProp_GradKernelLinear(int limit, float* w, float* dw, float* vt, float* bw, float* bdw, float* bvt, int bn, float lr);
void RunForTensor(float* w, float* dw, float* vt, int n, float* bw, float* bdw, float* bvt, int bn, float lr);
#endif // __NVCC__

#ifdef __cplusplus
}
#endif

#endif