#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include "Model.h"
#include "Optimizer.h"

int main() 
{
	Model m = Model_CreateGPU();
	Layer *l = Model_AddLayer(&m, Input_CreateGPU({ 128,128,3 }));
	l = Model_AddLayer(&m, Dense_CreateGPU(10, l));
	l = Model_AddLayer(&m, Dense_CreateGPU(10, l));
	l = Model_AddLayer(&m, Dense_CreateGPU(2, l));
	return 0;
}