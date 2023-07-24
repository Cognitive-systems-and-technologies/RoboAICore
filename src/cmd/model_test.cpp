#include <stdlib.h>
#include <stdio.h>
#include "Model.h"
#include "Utils.h"

int main()
{
	shape input = { 128,128,1 };
	Tensor x = Tensor_Create(input, 1.f);

	printf("Create model structure:\n");
	Model n = Model_Create();
	Layer* l = Model_AddLayer(&n, Input_Create(input));
	l = Model_AddLayer(&n, Conv2d_Create(96, { 11,11 }, { 2,2 }, 0, R_HE, l));
	l = Model_AddLayer(&n, Relu_Create(l));
	l = Model_AddLayer(&n, MaxPool2d_Create({ 5,5 }, { 2,2 }, 0, l));
	l = Model_AddLayer(&n, Conv2d_Create(64, { 3,3 }, { 1,1 }, 0, R_HE, l));
	l = Model_AddLayer(&n, Relu_Create(l));
	l = Model_AddLayer(&n, MaxPool2d_Create({ 3,3 }, { 1,1 }, 0, l));
	l = Model_AddLayer(&n, Conv2d_Create(32, { 3,3 }, { 1,1 }, 0, R_HE, l));
	l = Model_AddLayer(&n, Relu_Create(l));
	l = Model_AddLayer(&n, Conv2d_Create(32, { 3,3 }, { 1,1 }, 0, R_HE, l));
	l = Model_AddLayer(&n, Relu_Create(l));
	l = Model_AddLayer(&n, Conv2d_Create(32, { 3,3 }, { 1,1 }, 0, R_HE, l));
	l = Model_AddLayer(&n, Relu_Create(l));
	l = Model_AddLayer(&n, Conv2d_Create(32, { 3,3 }, { 1,1 }, 0, R_HE, l));
	l = Model_AddLayer(&n, Relu_Create(l));
	l = Model_AddLayer(&n, MaxPool2d_Create({ 3,3 }, { 1,1 }, 0, l));
	l = Model_AddLayer(&n, Dense_Create(4096, R_HE, l));
	l = Model_AddLayer(&n, Relu_Create(l));
	l = Model_AddLayer(&n, Dense_Create(4096, R_HE, l));
	l = Model_AddLayer(&n, Relu_Create(l));
	l = Model_AddLayer(&n, Dense_Create(3, R_XAVIER, l));

	printf("\nTest model forward pass:");
	Tensor* y = n.NetForward(&n, &x);
	PrintArray(y->w, y->n);
	printf("\nPress enter to close...");
	getchar();
	return 0;
}