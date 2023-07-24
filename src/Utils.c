#include "Utils.h"

void WriteToFile(const char* txt, const char* file)
{
	FILE* fptr;
	fptr = fopen(file, "w");

	if (fptr == NULL)
	{
		printf("Error!");
		exit(1);
	}
	fprintf(fptr, txt);
	fclose(fptr);
}

void PrintArray(float* w, int n)
{
	printf("\n[");
	for (size_t i = 0; i < n; i++)
	{
		printf("%f, ", w[i]);
	}
	printf("]\n");
}

void FillArray(float* w, int n, float v) 
{
	for (int i = 0; i < n; i++)
	{
		w[i] = v;
	}
}
