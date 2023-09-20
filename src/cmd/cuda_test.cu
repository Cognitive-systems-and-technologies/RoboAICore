#include <stdlib.h>
#include <stdio.h>
#include "Model.h"
#include "Optimizer.h"
#include "Losses.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Пример создания и обучения модели из трех полносвязных слоев и функциями активации гиперболического тангенса на GPU
int main() 
{
    //Определяем размерность входных данных
	shape input = { 128,128,1 };
    //Инициализация модели на GPU
	Model m = Model_CreateGPU();
    //Добавляем входной слой GPU к модели m, сохраняем ссылку на входной слой в inp
	Layer *inp = Model_AddLayer(&m, Input_CreateGPU(input));
    //Добавляем полносвязный слой GPU к модели m
	Layer* l = Model_AddLayer(&m, Dense_CreateGPU(128, inp));
    //Добавляем слой активации tanh GPU к модели m
	l = Model_AddLayer(&m, TanhA_CreateGPU(l));
	l = Model_AddLayer(&m, Dense_CreateGPU(128, l));
	l = Model_AddLayer(&m, TanhA_CreateGPU(l));
	l = Model_AddLayer(&m, Dense_CreateGPU(2, l));

	//Тест прямого прохода модели до оптимизации:
	printf("\nTest model forward pass:\n");
    //Задаем вход модели:
    Tensor test = Tensor_CreateGPU(input, 1.f);
	inp->input = &test;
    //Выполняем прямой проход модели:
	Model_ForwardGPU(&m);
    //Результат выполнения будет записан в выходном слое модели
    //Вывод тензора на консоль
	Tensor_PrintGPU(&l->output);

	//Инициализируем параметры оптимизации:
	OptParams p = OptParams_Create();
    //Задаем скорость обучения
	p.learning_rate = 0.001f;
    //Задаем метод оптимизации
	p.method = NRMSPROP;
    //Подготовка модели m для обучения на GPU с параметрами p
	PrepareTDataGPU(&m, &p);
    
    //Создаем два образа на которые будем обучать модель
	Tensor x1 = Tensor_CreateGPU(input, 1.f);
	Tensor x2 = Tensor_CreateGPU(input, -1.f);
    
    //Создаем два тензора y1 и y2 с ожидаемыми выходами модели для x1 и x2
	float data1[2] = {1.f, 0.f};
	float data2[2] = { 0.f, 1.f };
	Tensor y1 = Tensor_FromDataGPU({ 1,1,2 }, data1);
	Tensor y2 = Tensor_FromDataGPU({ 1,1,2 }, data2);

    //Цикл оптимизации из 300 шагов
	for (size_t i = 0; i < 300; i++)
	{
		//Шаг обучения для первого образа
        //Задаем вход модели:
		inp->input = &x1;
        //Выполняем прямой проход модели:
		Model_ForwardGPU(&m);
        //Вычисляем ошибку для текущего выхода модели:
		float loss1 = MSE_LossGPU(&l->output, &y1);
        //Выполняем обратный проход модели и вычисляем градиенты:
		Model_BackwardGPU(&m);
        //Выполняем шаг оптимизации и применяем градиенты к весовым коэффициентам модели:
		OptimizeModelGPU(&m, &p);

		//Шаг обучения для второго образа
		inp->input = &x2;
		Model_ForwardGPU(&m);
		float loss2 = MSE_LossGPU(&l->output, &y2);
		Model_BackwardGPU(&m);
		OptimizeModelGPU(&m, &p);

        //Средняя ошибка для двух образов на текущем шаге оптимизации
		float total_loss = (loss1 + loss2) * 0.5f;
		printf("loss: %f\n", total_loss);
	}

	printf("\nTest model forward pass:\n");
    //Тестирование модели после оптимизации
    //Тестирование выхода модели для первого образа
	inp->input = &x1;
	Model_ForwardGPU(&m);
	Tensor_PrintGPU(&l->output);
	
    //Тестирование выхода модели для второго образа
    inp->input = &x2;
	Model_ForwardGPU(&m);
	Tensor_PrintGPU(&l->output);

	printf("\nPress enter to close...");
	getchar();
	return 0;
}