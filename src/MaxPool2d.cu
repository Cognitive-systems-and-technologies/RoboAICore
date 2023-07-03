#include "MaxPool2d.h"
#include <stdlib.h>

#ifdef __NVCC__
Layer* MaxPool2d_CreateGPU(shape2 k_size, shape2 stride, int pad, Layer* in) 
{
	//input shape depth must be == 1
	Layer* l = (Layer*)malloc(sizeof(Layer));
	if (!l)
	{
		printf("MaxPool2d allocation error!");
		return NULL;
	}
	l->type = LT_MAXPOOL;
	int inn = in->out_shape.w * in->out_shape.h * in->out_shape.d;
	//calculate output shape
	l->out_shape.d = in->out_shape.d;
	l->out_shape.w = (int)((in->out_shape.w - k_size.w + pad * 2) / stride.w + 1);
	l->out_shape.h = (int)((in->out_shape.h - k_size.h + pad * 2) / stride.h + 1);
	printf("MaxPool2d output shape: [%d, %d, %d]\n", l->out_shape.w, l->out_shape.h, l->out_shape.d);

	l->n_inputs = inn;
	l->output = Tensor_CreateGPU(l->out_shape, 0);
	l->input = &in->output;

	float bias = 0.0f;
	MaxPool2d* ld = (MaxPool2d*)malloc(sizeof(MaxPool2d));
	if (ld) {
		ld->pad = pad;
		ld->stride.w = stride.w; ld->stride.h = stride.h;
		ld->k_size.w = k_size.w; ld->k_size.h = k_size.h;
	}
	else printf("MaxPool2d data allocation error\n");
	l->aData = ld;
	return l;
}

__global__ void MaxPool2d_ForwardKernels(shape limit, float* xw, float* outw, shape ishape, shape oshape, shape2 k_size, shape2 stride, int pad)
{
	int w = (blockIdx.x * blockDim.x) + threadIdx.x;
	int h = (blockIdx.y * blockDim.y) + threadIdx.y;
	int d = (blockIdx.z * blockDim.z) + threadIdx.z;
	if (w < limit.w && h < limit.h && d < limit.d) {

		float maxk = -FLT_MAX;
		for (size_t kh = 0; kh < k_size.h; kh++)
		{
			int cury = (h * stride.h - pad) + kh;
			for (size_t kw = 0; kw < k_size.w; kw++)
			{
				int curx = (w * stride.w - pad) + kw;
				if (curx >= 0 && cury >= 0 && curx < ishape.w && cury < ishape.h)
				{
					int xwi = ((ishape.w * cury) + curx) * ishape.d + d;
					float val = xw[xwi];
					if (val > maxk) maxk = val;
				}
			}
		}
		int owi = ((oshape.w * h) + w) * oshape.d + d;
		outw[owi] = maxk;
	}
}

Tensor* MaxPool2d_ForwardGPU(Layer* l) 
{
	MaxPool2d* data = (MaxPool2d*)l->aData;

	int w = l->out_shape.w;
	int h = l->out_shape.h;
	int d = l->out_shape.d;

	int threadsPerBlockX = 4;
	int threadsPerBlockY = 4;
	int threadsPerBlockZ = 64;

	dim3 gridDim((int)ceil(w / (float)threadsPerBlockX), (int)ceil(h / (float)threadsPerBlockY), (int)ceil(d / (float)threadsPerBlockZ));
	dim3 blockDim(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);

	MaxPool2d_ForwardKernels KERNEL_CALL(gridDim, blockDim) ({w,h,d},
		l->input->w,
		l->output.w, 
		l->input->s, 
		l->output.s,
		data->k_size, 
		data->stride, 
		data->pad);
	cudaDeviceSynchronize();
	return &l->output;
}

__global__ void MaxPool2d_BackwardKernels(shape limit, float* xw, float* xdw, float* outdw, shape ishape, shape oshape, shape2 k_size, shape2 stride, int pad)
{
	int w = (blockIdx.x * blockDim.x) + threadIdx.x;
	int h = (blockIdx.y * blockDim.y) + threadIdx.y;
	int d = (blockIdx.z * blockDim.z) + threadIdx.z;

	if (w < limit.w && h < limit.h && d < limit.d) {

		float maxk = -FLT_MAX;
		int khm = 0, kwm = 0;
		for (size_t kh = 0; kh < k_size.h; kh++)
		{
			int cury = (h * stride.h - pad) + kh;
			for (size_t kw = 0; kw < k_size.w; kw++)
			{
				int curx = (w * stride.w - pad) + kw;
				if (curx >= 0 && cury >= 0 && curx < ishape.w && cury < ishape.h)
				{
					int xwi = ((ishape.w * cury) + curx) * ishape.d + d;
					float val = xw[xwi];
					if (val > maxk) { maxk = val; kwm = curx; khm = cury; }
				}
			}
		}
		int odwi = ((oshape.w * h) + w) * oshape.d + d;
		float chain_grad = outdw[odwi];
		///-----------------------
		int id = ((ishape.w * khm) + kwm) * ishape.d + d;
		//xdw[id] += chain_grad;
		atomicAdd(&xdw[id], chain_grad);
	}
}

void MaxPool2d_BackwardGPU(Layer* l) 
{
	MaxPool2d* data = (MaxPool2d*)l->aData;

	int w = l->out_shape.w;
	int h = l->out_shape.h;
	int d = l->out_shape.d;

	int threadsPerBlockX = 4;
	int threadsPerBlockY = 4;
	int threadsPerBlockZ = 64;

	dim3 gridDim((int)ceil(w / (float)threadsPerBlockX), (int)ceil(h / (float)threadsPerBlockY), (int)ceil(d / (float)threadsPerBlockZ));
	dim3 blockDim(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);

	MaxPool2d_BackwardKernels KERNEL_CALL(gridDim, blockDim) ({w,h,d},
		l->input->w,
		l->input->dw,
		l->output.dw,
		l->input->s,
		l->output.s,
		data->k_size,
		data->stride,
		data->pad);
	cudaDeviceSynchronize();
}
#endif // __NVCC__
