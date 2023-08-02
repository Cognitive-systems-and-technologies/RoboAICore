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
	Layer *out = Model_AddLayer(&n, Dense_Create(2, R_XAVIER, l));
	
	//====optimization=====
	OptParams p = OptParams_Create();
	p.learning_rate = 0.01f;
	p.method = ADAN;

	printf("Optimizer: ADAN, lr = 0.01, loss = cross_entropy\n");
	for (size_t i = 0; i < 50; i++)
	{
		//first sample
		inp->input = &x1;
		Model_Forward(&n);
		float loss1 = Cross_entropy_Loss(&out->output, 0);
		Model_Backward(&n);
		OptimizeModel(&n, &p);

		//second sample
		inp->input = &x2;
		Model_Forward(&n);
		float loss2 = Cross_entropy_Loss(&out->output, 1);
		Model_Backward(&n);
		OptimizeModel(&n, &p);

		float total_loss = (loss1 + loss2) * 0.5f;
		printf("loss: %f\n", total_loss);
	}
	printf("\nTest model forward pass:");

	inp->input = &x1;
	Model_Forward(&n);
	Tensor o1 = SoftmaxProb(&out->output);
	PrintArray(o1.w, o1.n);
	Tensor_Free(&o1);

	inp->input = &x2;
	Model_Forward(&n);
	Tensor o2 = SoftmaxProb(&out->output);
	PrintArray(o2.w, o2.n);
	Tensor_Free(&o2);

	printf("\nPress enter to close...");
	getchar();
	return 0;
}