# RoboAI Core Library
Кросплатформенная библиотека глубокого обучения, для мобильных роботов и ПК.
Пример обучения агента на базе stm32f429. Мониторинг и управление агентом осуществляется при помощи веб интерфейса [NeuralInterface](https://github.com/Cognitive-systems-and-technologies/NeuralInterface)

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/12a71c56-2717-4aa6-8c5e-a0d0802e9ae2

Пример работы обученного агента:

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/b6212315-1973-41cf-9fdd-ebefeb0ca5e2

Пример компиляции программы для stm32:

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/9dda2685-c922-4ba5-9f9c-bd272eaf76bc

### Функции:
- Нейронные сети прямого распространения,
- Свёрточные нейронные сети,
- Алгоритмы обучения с подкреплением (deep RL)

### Адгоритмы оптимизации:
- SGD,
- Adagrad,
- RMSProp,
- Nesterov,
- Adam,
- Adan (Adaptive Nesterov Momentum Algorithm)

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

Далее представлен процесс сборки на примере использования cmake-gui + Visual Studio MSVC:
- скопируйте репозиторий
```bash
 $ git clone --recursive https://github.com/Cognitive-systems-and-technologies/RoboAICore.git
```
- откройте cmake и укажите пути к папке с проектом и папке, в которую будет собран проект, затем нажмите "configure" и выберете тип проекта и компилятор (для VS можно оставить по умолчанию),
- после завершения конфигурации нажмите "generate" и "open project". Проект откроется в VIsual Studio,
- в окне Visual Studio выберите тип компиляции и в меню "Build"->"Rebuild solution".

https://github.com/Cognitive-systems-and-technologies/RoboAICore/assets/100981393/3f293dad-f79e-4311-9cd7-68e277f902dc

После компиляции будут созданы исполняемые файлы с примерами создания и обучения моделей. Файлы с кодом примеров расположены в папке cmd:
[Examples](https://github.com/Cognitive-systems-and-technologies/RoboAICore/tree/main/src/cmd)

## Ресурсы:
- Описание функций библиотеки [RoboAICore API](https://github.com/Cognitive-systems-and-technologies/materials/blob/main/RAI_API.pdf).
