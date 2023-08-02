#include <stdlib.h>
#include <stdio.h>
#include "Model.h"
#include "Optimizer.h"
#include "Losses.h"
#include "Utils.h"

int main()
{
	shape input = { 128,128,1 };
	Tensor x1 = Tensor_Create(input, 1.f);
	Tensor x2 = Tensor_Create(input, -1.f);

	printf("Create model structure:\n");
	Model n = Model_Create();
	Layer* inp = Model_AddLayer(&n, Input_Create(input));
	Layer *l = Model_AddLayer(&n, Dense_Create(128, R_XAVIER, inp));
	l = Model_AddLayer(&n, TanhA_Create(l));
	l = Model_AddLayer(&n, Dense_Create(64, R_XAVIER, l));
	l = Model_AddLayer(&n, TanhA_Create(l));

	Layer *out1 = Model_AddLayer(&n, Dense_Create(32, R_XAVIER, l));
	out1 = Model_AddLayer(&n, TanhA_Create(out1));
	out1 = Model_AddLayer(&n, Dense_Create(2, R_XAVIER, out1));

	Layer* out2 = Model_AddLayer(&n, Dense_Create(32, R_XAVIER, l));
	out2 = Model_AddLayer(&n, TanhA_Create(out2));
	out2 = Model_AddLayer(&n, Dense_Create(2, R_XAVIER, out2));
	
	Tensor t1 = Tensor_Create(out2->out_shape, 0); t1.w[0] = 10.f;
	Tensor t2 = Tensor_Create(out2->out_shape, 0); t2.w[1] = 10.f;

	//====optimization=====
	OptParams p = OptParams_Create();
	p.learning_rate = 0.01f;
	p.method = ADAN;

	printf("Optimizer: ADAN, lr = 0.01, loss1 = cross_entropy, loss2 = mse\n");
	for (size_t i = 0; i < 500; i++)
	{
		//first sample
		inp->input = &x1;
		Model_Forward(&n);
		float loss1_1 = Cross_entropy_Loss(&out1->output, 0);
		float loss1_2 = MSE_Loss(&out2->output, &t1);
		Model_Backward(&n);
		OptimizeModel(&n, &p);

		//second sample
		inp->input = &x2;
		Model_Forward(&n);
		float loss2_1 = Cross_entropy_Loss(&out1->output, 1);
		float loss2_2 = MSE_Loss(&out2->output, &t2);
		Model_Backward(&n);
		OptimizeModel(&n, &p);

		float total_loss1 = (loss1_1 + loss2_1) * 0.5f;
		float total_loss2 = (loss1_2 + loss2_2) * 0.5f;
		printf("loss_1: %f, loss_2: %f\n", total_loss1, total_loss2);
	}

	printf("\nTest model forward pass:\n");
	printf("\nSample 1:");
	inp->input = &x1;
	Model_Forward(&n);
	Tensor o1 = SoftmaxProb(&out1->output);
	PrintArray(o1.w, o1.n);
	PrintArray(out2->output.w, out2->output.n);
	Tensor_Free(&o1);

	printf("\nSample 2:");
	inp->input = &x2;
	Model_Forward(&n);
	Tensor o2 = SoftmaxProb(&out1->output);
	PrintArray(o2.w, o2.n);
	PrintArray(out2->output.w, out2->output.n);
	Tensor_Free(&o2);

	printf("\nPress enter to close...");
	getchar();
	return 0;
}