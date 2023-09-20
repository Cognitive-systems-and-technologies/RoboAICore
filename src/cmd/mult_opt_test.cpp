#include <stdlib.h>
#include <stdio.h>
#include "Model.h"
#include "Optimizer.h"
#include "Losses.h"
#include "Utils.h"

//Пример создания и обучения модели с несколькими выходами на CPU
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

    //Создаем первую дополнительную ветку от слоя tanh модели с выходом out1
	Layer *out1 = Model_AddLayer(&n, Dense_Create(32, R_XAVIER, l));
	out1 = Model_AddLayer(&n, TanhA_Create(out1));
	out1 = Model_AddLayer(&n, Dense_Create(2, R_XAVIER, out1));

    //Создаем вторую дополнительную ветку от слоя tanh модели с выходом out2
	Layer* out2 = Model_AddLayer(&n, Dense_Create(32, R_XAVIER, l));
	out2 = Model_AddLayer(&n, TanhA_Create(out2));
	out2 = Model_AddLayer(&n, Dense_Create(2, R_XAVIER, out2));
	
    //Создаем два тензора t1 и t2 с ожидаемыми выходами второй ветки(out2) модели для x1 и x2
	Tensor t1 = Tensor_Create(out2->out_shape, 0); t1.w[0] = 10.f;
	Tensor t2 = Tensor_Create(out2->out_shape, 0); t2.w[1] = 10.f;

	//====optimization=====
    //Инициализируем параметры оптимизации:
	OptParams p = OptParams_Create();
    //Задаем скорость обучения
	p.learning_rate = 0.01f;
    //Задаем метод оптимизации
	p.method = ADAN;

	printf("Optimizer: ADAN, lr = 0.01, loss1 = cross_entropy, loss2 = mse\n");
    //Цикл оптимизации из 500 шагов
	for (size_t i = 0; i < 500; i++)
	{
		//Шаг обучения для первого образа
        //Задаем вход модели:
		inp->input = &x1;
        //Выполняем прямой проход модели:
		Model_Forward(&n);
        //Вычисляем ошибки для выходов модели out1 и out2:
		float loss1_1 = Cross_entropy_Loss(&out1->output, 0);
		float loss1_2 = MSE_Loss(&out2->output, &t1);
        //Выполняем обратный проход модели и вычисляем градиенты:
		Model_Backward(&n);
        //Выполняем шаг оптимизации и применяем градиенты к весовым коэффициентам модели:
		OptimizeModel(&n, &p);

		//Шаг обучения для второго образа
		inp->input = &x2;
		Model_Forward(&n);
		float loss2_1 = Cross_entropy_Loss(&out1->output, 1);
		float loss2_2 = MSE_Loss(&out2->output, &t2);
		Model_Backward(&n);
		OptimizeModel(&n, &p);

        //Средняя ошибка для двух образов на текущем шаге оптимизации отдельно для выходов out1 и out2
		float total_loss1 = (loss1_1 + loss2_1) * 0.5f;
		float total_loss2 = (loss1_2 + loss2_2) * 0.5f;
		printf("loss_1: %f, loss_2: %f\n", total_loss1, total_loss2);
	}

	printf("\nTest model forward pass:\n");
	printf("\nSample 1:");
    //Тестирование модели после оптимизации
    //Тестирование выхода модели для первого образа
	inp->input = &x1;
	Model_Forward(&n);
    //Так как использовалась Cross_entropy_Loss для первого выхода сети, применяем операцию softmax к выходу сети out1, для значений результата в диапазоне [0,1]
	Tensor o1 = SoftmaxProb(&out1->output);
	PrintArray(o1.w, o1.n);
	PrintArray(out2->output.w, out2->output.n);
    //Очищаем память для тензора с результатами операции softmax, т.к. он больше не нужен
	Tensor_Free(&o1);

	printf("\nSample 2:");
    //Тестирование выхода модели для второго образа
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