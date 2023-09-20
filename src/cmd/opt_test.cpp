#include <stdlib.h>
#include <stdio.h>
#include "Model.h"
#include "Optimizer.h"
#include "Losses.h"
#include "Utils.h"

//Пример создания и обучения модели из трех полносвязных слоев и функциями активации гиперболического тангенса на CPU
int main()
{
    //Определяем размерность входных данных
	shape input = { 128,128,1 };
    //Создаем два образа на которые будем обучать модель
	Tensor x1 = Tensor_Create(input, 1.f);
	Tensor x2 = Tensor_Create(input, -1.f);

	printf("Create model structure:\n");
    //Инициализация модели на CPU
	Model n = Model_Create();
    //Добавляем входной слой CPU к модели n, сохраняем ссылку на входной слой в inp
	Layer* inp = Model_AddLayer(&n, Input_Create(input));
    //Добавляем полносвязный слой CPU к модели n
	Layer *l = Model_AddLayer(&n, Dense_Create(128, R_XAVIER, inp));
    //Добавляем слой активации tanh CPU к модели n
	l = Model_AddLayer(&n, TanhA_Create(l));
	l = Model_AddLayer(&n, Dense_Create(64, R_XAVIER, l));
	l = Model_AddLayer(&n, TanhA_Create(l));
	Layer *out = Model_AddLayer(&n, Dense_Create(2, R_XAVIER, l));
	
	//====optimization=====
    //Инициализируем параметры оптимизации:
	OptParams p = OptParams_Create();
    //Задаем скорость обучения
	p.learning_rate = 0.01f;
    //Задаем метод оптимизации
	p.method = ADAN;

	printf("Optimizer: ADAN, lr = 0.01, loss = cross_entropy\n");
    //Цикл оптимизации из 50 шагов
	for (size_t i = 0; i < 50; i++)
	{
		//Шаг обучения для первого образа
        //Задаем вход модели:
		inp->input = &x1;
        //Выполняем прямой проход модели:
		Model_Forward(&n);
        //Вычисляем ошибку для текущего выхода модели:
		float loss1 = Cross_entropy_Loss(&out->output, 0);
        //Выполняем обратный проход модели и вычисляем градиенты:
		Model_Backward(&n);
        //Выполняем шаг оптимизации и применяем градиенты к весовым коэффициентам модели:
		OptimizeModel(&n, &p);

		//Шаг обучения для второго образа
		inp->input = &x2;
		Model_Forward(&n);
		float loss2 = Cross_entropy_Loss(&out->output, 1);
		Model_Backward(&n);
		OptimizeModel(&n, &p);

        //Средняя ошибка для двух образов на текущем шаге оптимизации
		float total_loss = (loss1 + loss2) * 0.5f;
		printf("loss: %f\n", total_loss);
	}
	printf("\nTest model forward pass:");
    //Тестирование модели после оптимизации
    //Тестирование выхода модели для первого образа
	inp->input = &x1;
	Model_Forward(&n);
    //Так как использовалась Cross_entropy_Loss применяем операцию softmax к выходу сети, для значений результата в диапазоне [0,1]
	Tensor o1 = SoftmaxProb(&out->output);
	PrintArray(o1.w, o1.n);
    //Очищаем память для тензора с результатами операции softmax, т.к. он больше не нужен
	Tensor_Free(&o1);
    
    //Тестирование выхода модели для второго образа
	inp->input = &x2;
	Model_Forward(&n);
	Tensor o2 = SoftmaxProb(&out->output);
	PrintArray(o2.w, o2.n);
	Tensor_Free(&o2);

	printf("\nPress enter to close...");
	getchar();
	return 0;
}