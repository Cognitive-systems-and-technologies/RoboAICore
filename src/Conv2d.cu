#include "Conv2d.h"
#include <stdlib.h>

#ifdef __NVCC__
Layer* Conv2d_CreateGPU(int num_kernels, shape2 k_size, shape2 stride, int pad, Layer* in) 
{
	//input shape depth must be == 1
	Layer* l = (Layer*)malloc(sizeof(Layer));
	if (!l)
	{
		printf("Conv2d allocation error!");
		return NULL;
	}
	l->type = LT_CONV;
	int inn = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	//calculate output shape
	l->out_shape.d = num_kernels;
	l->out_shape.w = (int)((in->out_shape.w - k_size.w + pad * 2) / stride.w + 1);
	l->out_shape.h = (int)((in->out_shape.h - k_size.h + pad * 2) / stride.h + 1);
	printf("Conv2d_GPU, output shape: [%d, %d, %d] pad: %d\n", l->out_shape.w, l->out_shape.h, l->out_shape.d, pad);

	l->n_inputs = inn;
	l->output = Tensor_CreateGPU(l->out_shape, 0);
	l->input = &in->output;

	float bias = 0.0f;
	Conv2dGPU* ld = (Conv2dGPU*)malloc(sizeof(Conv2dGPU));
	if (ld) {
		ld->pad = pad;
		ld->stride.w = stride.w; ld->stride.h = stride.h;
		ld->k_size.w = k_size.w; ld->k_size.h = k_size.h;

		shape4 ks = { k_size.w, k_size.h, in->out_shape.d, num_kernels };
		ld->kernels = Tensor4_CreateGPU(ks, 1.f); //assume that n input channels = 1 for now
		Tensor_Xavier_RandGPU(ld->kernels.w, ld->kernels.n);
			
		ld->biases = Tensor_CreateGPU({ 1, 1, num_kernels }, bias);
	}
	else printf("Conv2d data allocation error\n");
	l->aData = ld;
	return l;
}

__global__ void Conv2d_ForwardKernels(shape limit, float* xw, float* kerw, float* bw, float* outw, shape ishape, shape4 kshape, shape oshape, shape2 k_size, shape2 stride, int pad)
{
	int w = (blockIdx.x * blockDim.x) + threadIdx.x;
	int h = (blockIdx.y * blockDim.y) + threadIdx.y;
	int d = (blockIdx.z * blockDim.z) + threadIdx.z;
	if (w < limit.w && h < limit.h && d < limit.d) {

		float ksum = 0;
		for (int kh = 0; kh < k_size.h; kh++)
		{
			int cury = (h * stride.h - pad) + kh;
			for (int kw = 0; kw < k_size.w; kw++)
			{
				int curx = (w * stride.w - pad) + kw;
				for (int imd = 0; imd < ishape.d; imd++)
				{
					if (curx >= 0 && cury >= 0 && curx < ishape.w && cury < ishape.h)
					{
						int xwi = ((ishape.w * cury) + curx) * ishape.d + imd;
						int kwi = (((kshape.w * kh) + kw) * kshape.d + imd) * kshape.b + d;

						ksum += xw[xwi] * kerw[kwi];
					}
				}
			}
		}
		ksum += bw[d];
		int owi = ((oshape.w * h) + w) * oshape.d + d;
		outw[owi] = ksum;
		//printf("KSUM: %f\n", ksum);
	}
}

Tensor* Conv2d_ForwardGPU(Layer* l) 
{
	Conv2dGPU* data = (Conv2dGPU*)l->aData;
	//Tensor_CopyDataGPU(&l->output, &data->biases);
	//=====================
	int w = l->out_shape.w;
	int h = l->out_shape.h;
	int d = l->out_shape.d;

	int threadsPerBlockX = 4;
	int threadsPerBlockY = 4;
	int threadsPerBlockZ = 64;

	dim3 gridDim((int)ceil(w / (float)threadsPerBlockX), (int)ceil(h / (float)threadsPerBlockY), (int)ceil(d / (float)threadsPerBlockZ));
	dim3 blockDim(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);

	Conv2d_ForwardKernels KERNEL_CALL(gridDim, blockDim) ({w,h,d},
		l->input->w, data->kernels.w, data->biases.w,
		l->output.w, l->input->s, data->kernels.s, l->output.s, data->k_size, data->stride, data->pad);
	cudaDeviceSynchronize();
	return &l->output;
}

__global__ void Conv2d_BackwardKernels(shape limit, float* xw, float* xdw, float* kerw, float* kerdw, float* outdw, float* bdw, shape ishape, shape4 kshape, shape oshape, shape2 k_size, shape2 stride, int pad)
{
	int w = (blockIdx.x * blockDim.x) + threadIdx.x;
	int h = (blockIdx.y * blockDim.y) + threadIdx.y;
	int d = (blockIdx.z * blockDim.z) + threadIdx.z;
	if (w < limit.w && h < limit.h && d < limit.d) {

		int owi = ((oshape.w * h) + w) * oshape.d + d;
		float chain_grad = outdw[owi];

		for (int kh = 0; kh < k_size.h; kh++)
		{
			int cury = (h * stride.h - pad) + kh;
			for (int kw = 0; kw < k_size.w; kw++)
			{
				int curx = (w * stride.w - pad) + kw;
				for (int imd = 0; imd < ishape.d; imd++)
				{
					if (curx >= 0 && cury >= 0 && curx < ishape.w && cury < ishape.h)
					{
						int xwi = ((ishape.w * cury) + curx) * ishape.d + imd;
						int kwi = (((kshape.w * kh) + kw) * kshape.d + imd) * kshape.b + d;

						//kerdw[kwi] += xw[xwi] * chain_grad;
						atomicAdd(&kerdw[kwi], xw[xwi] * chain_grad);
						//xdw[xwi] += kerw[kwi] * chain_grad;
						float xdwi = kerw[kwi] * chain_grad;
						atomicAdd(&xdw[xwi], xdwi);
					}
				}
			}
		}

		//if(w==0&&h==0)
		//	bdw[d] += chain_grad;
		atomicAdd(&bdw[d], chain_grad);
	}
}

void Conv2d_BackwardGPU(Layer* l) 
{
	Conv2dGPU* data = (Conv2dGPU*)l->aData;
	//Tensor_CopyDataGPU(&l->output, &data->biases);
	//=====================
	int w = l->out_shape.w;
	int h = l->out_shape.h;
	int d = l->out_shape.d;

	int threadsPerBlockX = 4;
	int threadsPerBlockY = 4;
	int threadsPerBlockZ = 64;

	dim3 gridDim((int)ceil(w / (float)threadsPerBlockX), (int)ceil(h / (float)threadsPerBlockY), (int)ceil(d / (float)threadsPerBlockZ));
	dim3 blockDim(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);

	Conv2d_BackwardKernels KERNEL_CALL(gridDim, blockDim) ({w,h,d},
		l->input->w, l->input->dw, 
		data->kernels.w, data->kernels.dw, 
		l->output.dw,
		data->biases.dw,
		l->input->s, 
		data->kernels.s, 
		l->output.s, 
		data->k_size, 
		data->stride, 
		data->pad);
	cudaDeviceSynchronize();
}
#endif // __NVCC__
