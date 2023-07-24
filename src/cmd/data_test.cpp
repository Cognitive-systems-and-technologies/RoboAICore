#include <stdio.h>
#include <stdlib.h>
#include "Model.h"
#include "Optimizer.h"
#include "Utils.h"

void TensorTest() 
{
	Tensor t = Tensor_Create({ 5, 5, 3 }, 1.f);
	Tensor_Print(&t);
	Tensor_Free(&t);
}

//Dense cpu test
void DenseTest()
{
	Tensor x = Tensor_Create({ 5, 5, 3 }, 2.f);
	Layer* inp = Input_Create(x.s);
	Layer* de = Dense_Create(10, R_XAVIER, inp);

	Input_Forward(inp, &x);
	Dense_Forward(de);
	//forward test
	PrintArray(de->output.w, de->output.n);
	//backward test
	FillArray(de->output.dw, de->output.n, 2.f);
	Dense_Backward(de);
	Dense* data = (Dense*)de->aData;
	PrintArray(data->kernels[0].dw, data->kernels[0].n);
	PrintArray(data->biases.dw, data->biases.n);

	Input_Free(inp);
	Dense_Free(de);
	Tensor_Free(&x);
}

//Conv cpu test
void ConvTest()
{
	Tensor x = Tensor_Create({ 10, 10, 3 }, 2.f);
	Layer* inp = Input_Create(x.s);
	Layer* conv = Conv2d_Create(10, { 3,3 }, {2,2}, 0, R_XAVIER, inp);

	Input_Forward(inp, &x);
	Conv2d_Forward(conv);
	//forward test
	PrintArray(conv->output.w, conv->output.n);
	//backward test
	FillArray(conv->output.dw, conv->output.n, 2.f);
	Conv2d_Backward(conv);
	Conv2d* data = (Conv2d*)conv->aData;
	PrintArray(data->kernels[0].dw, data->kernels[0].n);
	PrintArray(data->biases.dw, data->biases.n);

	Input_Free(inp);
	Conv2d_Free(conv);
	Tensor_Free(&x);
}



int main() 
{
	printf("Tensor creation test:\n");
	TensorTest();
	printf("\nDense layer test:\n");
	DenseTest();
	printf("\nConv2d layer test:\n");
	ConvTest();
	
	printf("\nPress enter to close...");
	getchar();
	return 0;
}