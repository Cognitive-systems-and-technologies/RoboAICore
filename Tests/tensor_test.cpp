#include "Tensor.h"

#include <stdlib.h>
#include <stdio.h>

void printTensor(Tensor* x)
{
	for (size_t d = 0; d < x->s.d; d++)
	{
		printf("[\n");
		for (size_t i = 0; i < x->s.w; i++)
		{
			for (size_t j = 0; j < x->s.h; j++)
			{
				printf("%f ", Tensor_Get(x, i, j, d));
			}
			printf("\n");
		}
		printf("]\n");
	}
}

void Tensor_ToJSON_Test() 
{
	Tensor* t = Tensor_Create({ 10,10,3 }, 2, 0);
	Tensor_Set(t, 0, 1, 0, 0.1234f);
	cJSON* node = Tensor_To_JSON(t);

	const char* txt = cJSON_Print(node);
	printf("Tensor: %s\n", txt);

	cJSON* w = cJSON_GetObjectItem(node, "w");

	//const char* jTensorStr = "{\"s\":[1,1,3],\"w\":[5,5,5],\"dw\":[0,0,0]}";//test str

	cJSON* jTensor = cJSON_Parse(txt);
	Tensor* jten = Tensor_From_JSON(jTensor);

	printTensor(jten);

	Tensor* t2 = Tensor_Create({ 10,10,3 }, 0, 0);
	Tensor_Load_JSON(t2, jTensor);
	printTensor(t2);

	cJSON_free(jTensor);
	cJSON_free(node);
	Tensor_Free(t);
	Tensor_Free(jten);
}

int main() 
{
	Tensor_ToJSON_Test();
	return 0;
}