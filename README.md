# RoboAI Core Library
A cross-platform deep learning library for mobile robots and PCs. The library can be used in areas where it is necessary to develop systems using neural networks, deep machine learning models, and reinforcement learning of intelligent agents.

The main areas of applied use of the library:
- educational robotics;
- mobile robots, cars, manipulators, etc.,
- smart home devices.

To illustrate the operation of the deep learning algorithms presented in the library, the task of training a mobile robot on the Yahboom Raspbot transport platform with ultrasonic rangefinders installed on the robot to determine distances to objects of the HC-SR04 type is shown. The robot also has a debug board [32f429 discovery](https://www.st.com/en/evaluation-tools/32f429idiscovery.html) under which it operates.

An example of training an agent based on stm32f429. Training is performed on the stm32f429 hardware using the DeepRl algorithm implemented in the library. Data on the training process is transmitted via the esp8266 module in the form of http messages to the server part for monitoring. Monitoring and control of the agent is performed using a specially developed web interface [NeuralInterface](https://github.com/Cognitive-systems-and-technologies/NeuralInterface)

The robot self-training task is formulated as follows:
The input vector (state vector S) takes values ​​from three HC-SR04 sensors and has the form: [d1, d2, d3], where d is the distance to obstacles. The output of the artificial neural network is the vector [a1, a2, a3], where a is the estimate of one of the three actions that the agent can perform. Actions a1-go straight, a2-turn left, a3-turn right. The reward function is as follows:
```
float GetReward(Eyes *eyes, int action, float max_length)
{
float proximity_reward = 0.0f;
proximity_reward += eyes->distVec[0] / max_length;
proximity_reward += eyes->distVec[1] / max_length;
proximity_reward += eyes->distVec[2] / max_length;
proximity_reward = proximity_reward / 3.f;
float forwardReward = 0;
if (action == 0 && proximity_reward > 0.8f) forwardReward = 0.5f;
float res = proximity_reward + forwardReward;
return res;
}
```
If there are no obstacles and action a1 is performed, the agent receives the maximum reward, otherwise the agent receives a reward depending on the distance to the obstacles, the smaller it is, the smaller the reward. The agent also receives additional rewards when it goes beyond the obstacle-bound zone. All structures and learning algorithms are executed on STM32F429. Only monitoring data and agent control signals are transmitted to the local PC. To ensure communication between the agent and the server, an ESP8266 module was used, which implements an http client for sending requests to the global server and a local web server for processing incoming requests. The robot is considered trained when, after a certain number of iterations, the learning error value is less than 1.f and the agent can bypass obstacles, moving towards the exit from the obstacle-bound zone.

Example of a trained agent:

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/ea0d8646-0c95-4f4a-b4c9-754df7526ee1

Example of the agent training process:

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/12a71c56-2717-4aa6-8c5e-a0d0802e9ae2

Example of program compilation for stm32f429:

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/9dda2685-c922-4ba5-9f9c-bd272eaf76bc

As a test of reinforcement learning algorithms, an example of finding the shortest paths to a goal in a maze (qmaze algorithm) was chosen. Let's say we have a 10X10 cell maze, some of the cells are free, some of them are closed - these are the walls of the maze. An agent is placed in the maze, which needs to get to the target cell in the minimum number of steps. The agent receives a small penalty for each move on a free cell and a larger penalty for trying to move to a closed cell. The reason for such a negative penalty is that we want the agent to get to the target cell along the shortest path and not crash into the walls. For moving to the target cell, the agent receives the maximum reward. The agent can only move along free cells, the main goal of the agent is to get to the target cell.

An example of the qmaze algorithm implemented using the RoboAICore library:

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/303822bc-dc65-43bc-8983-5437ec26f108

After a certain number of training episodes, the agent learned to determine the shortest distance to the target cell from the overwhelming majority of starting positions.

A convolutional neural network model for object recognition was tested for raspberry pi. As an example of recognition, the task of recognizing one agent by another when it enters the camera's field of view was chosen.

For training, a small dataset was compiled for 2 classes: room and agent. The network input has a dimension of 227x227x3. In total, 40 images were used for training, 20 for each class.

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/adc06d2f-c17e-45cb-9943-d3c8b28b5788

### Features:
- Feedforward neural networks,
- Convolutional neural networks,
- Reinforcement learning algorithms (deep RL)

### Optimization algorithms:
- SGD - simple stochastic gradient descent,
- Adagrad - adaptive gradient algorithm,
- RMSProp - root mean square propagation,
- Nesterov - Nesterov Accelerated Gradient,
- Adam - adaptive momentum estimation,
- Adan - Adaptive Nesterov Momentum Algorithm.

## Assembly and compilation:
The project was tested on: STM32f429, RaspberryPi 3 model B and a personal computer.

Requirements for working on microcontrollers:
- The library uses the float data type, so the microcontroller must support floating-point number calculations (FPU). For stm32, this is, for example, STM32F4xx, STM32F74x/5x, STM32L4xx, STM32F76x/7x, etc.
- The amount of RAM required for operation depends on the size of the model being created and can vary from several tens of kilobytes to the size of the memory available on the device.
- The size of programmable memory from 500 KB and above.
- Compiler support and the presence of standard C libraries (libc).

The project uses the CMake build system.
To build and compile the project on a PC, you need to have the cmake program and compiler installed.

To compile a library with support for GPU computing, you need to install the development tools for creating applications for the CUDA architecture - “NVIDIA GPU Computing Toolkit”. You can download the installation package from the nvidia website at the link:
```
https://developer.nvidia.com/cuda-downloads
```
GPUs with the compute_60 (Pascal) architecture version and higher are supported.

Below is the process of assembling and compiling for the Windows operating system using cmake-gui + Visual Studio MSVC as an example:
- copy the repository
```bash
$ git clone --recursive https://github.com/Cognitive-systems-and-technologies/RoboAICore.git
```
- open cmake and specify the paths to the project folder and the folder where the project will be built, then click "configure" and select the project type and compiler (for VS you can leave the default),
- after completing the configuration, click "generate" and "open project". The project will open in VIsual Studio,
- in the Visual Studio window, select the compilation type and in the menu "Build"->"Rebuild solution".

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/3f293dad-f79e-4311-9cd7-68e277f902dc

Here is the build process on RaspberryPi for Debian Linux operating system:
- copy the repository
- open terminal and run commands to update and install cmake
```
sudo apt update
sudo apt install -y cmake
```
- go to the source code folder and run the command line from the directory
- run the following commands to build the project and compile:
```
cmake .
make
```

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/a33fce38-b1c1-4e9d-a61a-3d87e6776f97

After compilation, executable files with examples of creating and training models will be created. Files with example code are located in the cmd folder:
[examples](https://github.com/Cognitive-systems-and-technologies/RoboAICore/tree/main/src/cmd). List of examples:
- cuda_test.cu - example of creating and training a model of three fully connected layers and hyperbolic tangent activation functions on the GPU,
- rand_test.cpp - example of the algorithms for initializing weight coefficients,
- opt_test.cpp - example of creating and training a model of three fully connected layers and hyperbolic tangent activation functions on the CPU,
- mult_opt_test.cpp - example of creating and training a model with multiple outputs on the CPU,
- model_test.cpp - example of creating a deep neural network model (like AlexNet) on the CPU,
- data_test.cpp - example of working with data and layer functions.

The following is the compilation process for [stm32f407](https://www.st.com/en/evaluation-tools/stm32f4discovery.html) in the CooCox CoIDE environment:

Configuring the environment and project:
- Download and install a set of software packages required for compiling and generating executable code - Arm GNU Toolchain:
```
https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads
```
- In the CooCox CoIDE environment, execute the commands Project -> Select toolchain path
- In the window that appears, specify the path to the bin folder of the installed Arm GNU Toolchain
- Create a new project and select the type of board or chip
- After completing the project setup, select the set of peripherals you want to work with, before standard C library component
- In the Configuration project settings, enable FPU support and in the Link category, select Use base C library

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/ca13f750-7fe4-4272-8e4a-f365609648aa

Example of project compilation and test on stm32f407:
- Create a new group (Add Group) in the project and add library files to it
- For comfortable work with data, you can increase the stack size from 512 bytes (default) to, for example, 32Kb. To do this, change the STACK_SIZE value to 0x00007D00 in the cmsis_boot/startup/startup_stm32f4xx.c project file (#define STACK_SIZE 0x00007D00)
- Select Project -> Rebuild

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/00fb1472-2094-40c8-a01c-5f9e6629c42d

## Resources:
- Description of the [RoboAICore API](https://github.com/Cognitive-systems-and-technologies/materials/blob/main/RAI_API.pdf) library functions.


---

# RoboAI Core Library
Кроссплатформенная библиотека глубокого обучения, для мобильных роботов и ПК. Библиотека может применяться в направлениях, где требуется разработка систем, использующих нейронные сети, модели глубокого машинного обучения, обучение с подкреплением интеллектуальных агентов. 

Основные направления прикладного использования библиотеки:
- образовательная робототехника;
- мобильные роботы, машинки, манипуляторы и т.п.,
- устройства систем умного дома.

Для иллюстрации работы алгоритмов глубокого обучения, представленных в библиотеке показана задача обучения мобильного робота на транспортной платформе Yahboom Raspbot с установленными на роботе ультразвуковыми дальномерами для определения расстояний до объектов типа HC-SR04. На робот также установлена отладочная плата [32f429 discovery](https://www.st.com/en/evaluation-tools/32f429idiscovery.html) под управлением которой он работает.

Пример обучения агента на базе stm32f429. Обучение происходит на аппаратной части stm32f429, используя алгоритм DeepRl реализованный в библиотеке. Данные о процессе обучения передаются при помощи модуля esp8266 в виде http-сообщений на серверную часть для мониторинга. Мониторинг и управление агентом осуществляется при помощи специально разработанного веб-интерфейса [NeuralInterface](https://github.com/Cognitive-systems-and-technologies/NeuralInterface)

Задача самообучения робота формулируется следующим образом:
Входной вектор (вектор состояния S) принимает значения с трех сенсоров HC-SR04 и имеет вид: [d1, d2, d3], где d – дистанция до препятствий. Выход искусственной нейронной сети – вектор [a1, a2, a3], где а – оценка одного из трех действий, которые может выполнять агент. Действия a1-ехать прямо, a2-повернуть налево, a3-повернуть направо. Функция поощрения имеет следующий вид:
```
float GetReward(Eyes *eyes, int action, float max_length)
{
	float proximity_reward = 0.0f;
	proximity_reward += eyes->distVec[0] / max_length;
	proximity_reward += eyes->distVec[1] / max_length;
	proximity_reward += eyes->distVec[2] / max_length;
	proximity_reward = proximity_reward / 3.f;
	float forwardReward = 0;
	if (action == 0 && proximity_reward > 0.8f) forwardReward = 0.5f;
	float res = proximity_reward + forwardReward;
	return res;
}
```
Если нет препятствий, и выполняется действие a1 - агент получает максимальное поощрение, в противном случае агент получает поощрение в зависимости от дистанции до препятствий, чем она меньше, тем меньше поощрение. Также агент получает дополнительное поощрение при выходе за пределы ограниченной препятствиями зоны. Все структуры и алгоритмы обучения выполняются на STM32F429. На локальный ПК передаются только данные для мониторинга и сигналы управления агентом. Для обеспечения связи между агентом и сервером был использован модуль ESP8266, на котором реализован http-клиент для отправки запросов на глобальный сервер и локальный web-сервер для обработки входящих запросов. Робот считается обученным, когда в процессе определенного количества итераций значение ошибки обучения меньше 1.f и агент может объезжать препятствия, двигаясь в сторону выхода из ограниченной препятствиями зоны.

Пример работы обученного агента:

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/ea0d8646-0c95-4f4a-b4c9-754df7526ee1

Пример процесса обучения агента:

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/12a71c56-2717-4aa6-8c5e-a0d0802e9ae2

Пример компиляции программы для stm32f429:

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/9dda2685-c922-4ba5-9f9c-bd272eaf76bc

В качестве тестирования алгоритмов обучения с подкреплением был выбран пример с поиском кратчайших путей к цели в лабиринте (алгоритм qmaze). Допустим, у нас есть лабиринт 10X10 ячеек, часть ячеек свободны, часть из них закрыты - это стенки лабиринта. В лабиринт помещается агент, которому нужно за минимальное количество шагов добраться до целевой ячейки. Агент получает небольшой штраф за каждый ход по свободной ячейке и больший штраф за попытку хода на закрытую ячейку. Причина такого отрицательного наказания в том, что мы хотим, чтобы агент попал в целевую ячейку по кратчайшему пути и не врезался в стены. За переход в целевую ячейку агент получает максимальное поощрение. Агент может передвигаться только по свободным клеткам, основная цель агента - добраться до целевой ячейки. 

Пример работы алгоритма qmaze реализованного с использованием библиотеки RoboAICore:

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/303822bc-dc65-43bc-8983-5437ec26f108

После некоторого количества эпизодов обучения агент научился определять кратчайшее расстояние до целевой ячейки из подавляющего числа стартовых позиций.

Для raspberry pi была протестирована модель сверточной нейросети для распознавания объектов. В качестве примера распознавания было выбрана задача распознавания одним агентом другого, когда тот попадает в поле зрения камеры. Для обучения был составлен небольшой датасет на 2 класса: комната и агент. Вход сети имеет размерность 227x227x3. Всего при обучении было использовано 40 изображений по 20 на каждый класс.

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/adc06d2f-c17e-45cb-9943-d3c8b28b5788

### Функции:
- Нейронные сети прямого распространения,
- Свёрточные нейронные сети,
- Алгоритмы обучения с подкреплением (deep RL)

### Адгоритмы оптимизации:
- SGD - метод простого стохастического градиентного спуска (stochastic gradient descent),
- Adagrad - метод адаптивного градиентного спуска (adaptive gradient algorithm),
- RMSProp - метод модифицированного адаптивного градиентного спуска (root mean square propagation),
- Nesterov - алгоритм Нестерова, метод накопления импульса (Nesterov Accelerated Gradient),
- Adam - метод адаптивной оценки момента (Adaptive Moment Estimation),
- Adan - адаптивный алгоритм импульса Нестерова (Adaptive Nesterov Momentum Algorithm).

## Сборка и компиляция:
Проект был протестирован на: STM32f429, RaspberryPi 3 model B и персональном компьютере.

Требования для работы на микроконтроллерах:
- В библиотеке используется тип данных float, поэтому необходима поддержка микроконтроллером вычислений чисел с плавающей точкой (FPU). Для stm32 это, например, STM32F4xx, STM32F74x/5x, STM32L4xx, STM32F76x/7x и т.п.
- Объем оперативной памяти, требуемой для работы, зависит от размеров создаваемой модели и может варьироваться от нескольких десятков килобайт и до размеров доступной на устройстве памяти.
- Размер программируемой памяти от 500Кб и выше.
- Поддержка компилятором и наличие стандартных библиотек С (libc).

Проект использует систему сборки CMake.
Для сборки и компиляции проекта на ПК необходимо наличие установленной программы cmake и компилятора.

Для компиляции библиотеки с поддержкой вычислений на GPU необходимо установить инструментальные средства разработки для создания приложений для архитектуры CUDA - “NVIDIA GPU Computing Toolkit”. Загрузить пакет для установки можно на сайте nvidia по ссылке:
```
https://developer.nvidia.com/cuda-downloads
```
Поддерживаются графические процессоры с версией архитектуры compute_60 (Pascal) и выше.

Далее представлен процесс сборки и компиляции для операционной системы Windows на примере использования cmake-gui + Visual Studio MSVC:
- скопируйте репозиторий
```bash
 $ git clone --recursive https://github.com/Cognitive-systems-and-technologies/RoboAICore.git
```
- откройте cmake и укажите пути к папке с проектом и папке, в которую будет собран проект, затем нажмите "configure" и выберете тип проекта и компилятор (для VS можно оставить по умолчанию),
- после завершения конфигурации нажмите "generate" и "open project". Проект откроется в VIsual Studio,
- в окне Visual Studio выберите тип компиляции и в меню "Build"->"Rebuild solution".

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/3f293dad-f79e-4311-9cd7-68e277f902dc

Далее представлен процесс сборки на RaspberryPi для операционной системы Debian Linux:
- скопируйте репозиторий
- откройте терминал и выполните команды для обновления и установки cmake
```
sudo apt update
sudo apt install -y cmake
```
- перейдите в папку с исходным кодом и запустите командную строку из дирректории
- выполните следующие команды для сборки проекта и компиляции:
```
cmake .
make
```

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/a33fce38-b1c1-4e9d-a61a-3d87e6776f97

После компиляции будут созданы исполняемые файлы с примерами создания и обучения моделей. Файлы с кодом примеров расположены в папке cmd:
[примеры](https://github.com/Cognitive-systems-and-technologies/RoboAICore/tree/main/src/cmd). Список примеров:
- cuda_test.cu - пример создания и обучения модели из трех полносвязных слоев и функциями активации гиперболического тангенса на GPU,
- rand_test.cpp - пример работы алгоритмов для инициализации весовых коэффициентов,
- opt_test.cpp - пример создания и обучения модели из трех полносвязных слоев и функциями активации гиперболического тангенса на CPU,
- mult_opt_test.cpp - пример создания и обучения модели с несколькими выходами на CPU,
- model_test.cpp - пример создания глубокой модели нейросети (по типу AlexNet) на CPU,
- data_test.cpp - пример работы с данными и функциями слоев.

Далее представлен процесс компиляции для [stm32f407](https://www.st.com/en/evaluation-tools/stm32f4discovery.html) в среде CooCox CoIDE:

Настройка среды и проекта:
- Загрузите и установите набор пакетов программ, необходимых для компиляции и генерации выполняемого кода - Arm GNU Toolchain:
```
https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads
```
- В среде CooCox CoIDE выполните команды Project -> Select toolchain path
- В появившемся окне укажите путь к папке bin установленного Arm GNU Toolchain
- Создайте новый проект и выберите тип платы или чипа
- После завершения настройки проекта выберите набор периферии с которой хотите работать, добавте компонент стандартных библиотек C
- В настройках проекта Configuration включите поддержку FPU и в категории Link выберите Use base C library

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/ca13f750-7fe4-4272-8e4a-f365609648aa

Пример компиляции проекта и тест на stm32f407:
- Создайте новую группу (Add Group) в проекте и добавте в нее файлы библиотеки
- Для комфортной работы с данными, можно увеличить размер стека с 512байт (по умолчанию) до, например, 32Кб. Для этого в файле проекта cmsis_boot/startup/startup_stm32f4xx.c необходимо изменить значение STACK_SIZE на 0x00007D00 (#define STACK_SIZE 0x00007D00)
- Выберите Project -> Rebuild

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/00fb1472-2094-40c8-a01c-5f9e6629c42d

## Ресурсы:
- Описание функций библиотеки [RoboAICore API](https://github.com/Cognitive-systems-and-technologies/materials/blob/main/RAI_API.pdf).
