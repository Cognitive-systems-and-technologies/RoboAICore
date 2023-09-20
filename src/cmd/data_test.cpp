#include <stdio.h>
#include <stdlib.h>
#include "Model.h"
#include "Optimizer.h"
#include "Utils.h"
#include "Losses.h"

//Пример работы с данными и функциями слоев
void TensorTest() 
{
    //Создание тензора размерностью 5x5x1 и значениями = 1.f
	Tensor t = Tensor_Create({ 5, 5, 1 }, 1.f);
    //Заполнение элементов массива w с количеством элементов n случайными значениями по непрерывному равномерному распределению.
	Tensor_Xavier_Rand(t.w, t.n);
    //Применение операции softmax к тензору t
	Tensor sm = SoftmaxProb(&t);
    //Вывод тензора на консоль
	Tensor_Print(&sm);
	//Очищение памяти для тензоров, т.к. они больше не нужны
	Tensor_Free(&sm);
	Tensor_Free(&t);
    //Создание тензора из массива data и размерностью 1x1x5
	float data[5] = { 1,2,3,4,5 };
	Tensor fromData = Tensor_FromData({ 1,1,5 }, data);
	Tensor_Print(&fromData);
	Tensor_Free(&fromData);
}

//Тест прямого и обратного проходов полносвязного слоя
void DenseTest()
{
	Tensor x = Tensor_Create({ 5, 5, 3 }, 2.f);
	Layer* inp = Input_Create(x.s);
	Layer* de = Dense_Create(10, R_XAVIER, inp);

	inp->input = &x;
	//Прямой проход
    Input_Forward(inp);
    Dense_Forward(de);
	PrintArray(de->output.w, de->output.n);
	//Обратный проход
	FillArray(de->output.dw, de->output.n, 2.f);
	Dense_Backward(de);
	Dense* data = (Dense*)de->aData;
	PrintArray(data->kernels[0].dw, data->kernels[0].n);
	PrintArray(data->biases.dw, data->biases.n);

	Input_Free(inp);
	Dense_Free(de);
	Tensor_Free(&x);
}

//Тест прямого и обратного проходов сверточного слоя
void ConvTest()
{
	Tensor x = Tensor_Create({ 10, 10, 3 }, 2.f);
	Layer* inp = Input_Create(x.s);
	Layer* conv = Conv2d_Create(10, { 3,3 }, {2,2}, 0, R_XAVIER, inp);

	inp->input = &x;
    //Прямой проход
	Input_Forward(inp);
	Conv2d_Forward(conv);
	PrintArray(conv->output.w, conv->output.n);
	//Обратный проход
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