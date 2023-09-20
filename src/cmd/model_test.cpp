#include <stdlib.h>
#include <stdio.h>
#include "Model.h"
#include "Utils.h"

//Пример создания глубокой модели нейросети (по типу AlexNet) на CPU
int main()
{
    //Определяем размерность входных данных
	shape input = { 128,128,1 };
    //Создаем тензор для теста прямого прохода модели
	Tensor x = Tensor_Create(input, 1.f);

	printf("Create model structure:\n");
    //Инициализация модели на CPU
	Model n = Model_Create();
    //Добавляем входной слой CPU к модели n, сохраняем ссылку на входной слой в inp
	Layer* inp = Model_AddLayer(&n, Input_Create(input));
    //Добавляем сверточный слой CPU к модели n
	Layer* l = Model_AddLayer(&n, Conv2d_Create(96, { 11,11 }, { 2,2 }, 0, R_HE, inp));
    //Добавляем слой активации Relu CPU к модели n
	l = Model_AddLayer(&n, Relu_Create(l));
    //Добавляем MaxPool слой CPU к модели n
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
    //Добавляем полносвязный слой CPU к модели n
	l = Model_AddLayer(&n, Dense_Create(4096, R_HE, l));
	l = Model_AddLayer(&n, Relu_Create(l));
	l = Model_AddLayer(&n, Dense_Create(4096, R_HE, l));
	l = Model_AddLayer(&n, Relu_Create(l));
	Layer* out = Model_AddLayer(&n, Dense_Create(3, R_XAVIER, l));

	printf("\nTest model forward pass:");
    //Тестирование прямого прохода модели
    //Задаем вход модели:
	inp->input = &x;
    //Выполняем прямой проход модели:
	Model_Forward(&n);
    //Вывод массива выходного тензора на консоль
	PrintArray(out->output.w, out->output.n);
	printf("\nPress enter to close...");
	getchar();
	return 0;
}