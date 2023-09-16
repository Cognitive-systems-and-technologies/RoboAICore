#include <stdlib.h>
#include <stdio.h>
#include "Model.h"
#include "Optimizer.h"
#include "Losses.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main() 
{
	shape input = { 128,128,1 };
	Tensor test = Tensor_CreateGPU(input, 1.f);
	Model m = Model_CreateGPU();
	Layer *inp = Model_AddLayer(&m, Input_CreateGPU(test.s));
	Layer* l = Model_AddLayer(&m, Dense_CreateGPU(128, inp));
	l = Model_AddLayer(&m, TanhA_CreateGPU(l));
	l = Model_AddLayer(&m, Dense_CreateGPU(128, l));
	l = Model_AddLayer(&m, TanhA_CreateGPU(l));
	l = Model_AddLayer(&m, Dense_CreateGPU(2, l));

	//test forward pass
	printf("\nTest model forward pass:\n");
	inp->input = &test;
	Model_ForwardGPU(&m);
	Tensor_PrintGPU(&l->output);

	//training
	OptParams p = OptParams_Create();
	p.learning_rate = 0.001f;
	p.method = NRMSPROP;
	PrepareTDataGPU(&m, &p);

	Tensor x1 = Tensor_CreateGPU(input, 1.f);
	Tensor x2 = Tensor_CreateGPU(input, -1.f);

	float data1[2] = {1.f, 0.f};
	float data2[2] = { 0.f, 1.f };
	Tensor y1 = Tensor_FromDataGPU({ 1,1,2 }, data1);
	Tensor y2 = Tensor_FromDataGPU({ 1,1,2 }, data2);

	for (size_t i = 0; i < 300; i++)
	{
		//first sample
		inp->input = &x1;
		Model_ForwardGPU(&m);
		float loss1 = MSE_LossGPU(&l->output, &y1);
		Model_BackwardGPU(&m);
		OptimizeModelGPU(&m, &p);

		//second sample
		inp->input = &x2;
		Model_ForwardGPU(&m);
		float loss2 = MSE_LossGPU(&l->output, &y2);
		Model_BackwardGPU(&m);
		OptimizeModelGPU(&m, &p);

		float total_loss = (loss1 + loss2) * 0.5f;
		printf("loss: %f\n", total_loss);
	}

	//test forward pass
	printf("\nTest model forward pass:\n");
	inp->input = &x1;
	Model_ForwardGPU(&m);
	Tensor_PrintGPU(&l->output);
	inp->input = &x2;
	Model_ForwardGPU(&m);
	Tensor_PrintGPU(&l->output);

	printf("\nPress enter to close...");
	getchar();
	return 0;
}