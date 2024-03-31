#include "Utils.h"

float* createFloatArray(int n) 
{
	float* a = (float*)malloc(sizeof(float)*n);
	if (!a) 
	{
		printf("Array allocation error\n");
		return NULL;
	}
	else {
		memset(a, 0, sizeof(float) * n);
		return a;
	}
}

int* createIntArray(int n)
{
	int* a = (int*)malloc(sizeof(int) * n);
	if (!a)
	{
		printf("Array allocation error\n");
		return NULL;
	}
	else {
		memset(a, 0, sizeof(int) * n);
		return a;
	}
}

void NormalizeArray(float *w, float n) 
{
	float stdev = StandardDeviation(w, n);
	float mean = 0, sum = 0, eps = 1e-10f;
	for (int i = 0; i < n; ++i) {
		sum += w[i];
	}
	mean = sum / (float)n;

	for (size_t i = 0; i < n; i++)
	{
		float norm = (w[i] - mean) / (stdev + eps);
		w[i] = norm;
	}
}

float StandardDeviation(float *data, int n) 
{
	float sum = 0.0f, mean, SD = 0.0f;
	int i;
	for (i = 0; i < n; ++i) {
		sum += data[i];
	}
	mean = sum / (float)n;
	for (i = 0; i < n; ++i) {
		float x = data[i] - mean;
		SD += x * x;
	}
	return sqrtf(SD / (float)n);
}

void FlipArray(float* w, int n) 
{
	for (size_t i = 0; i < n/2; i++)
	{
		float temp = w[i];
		w[i] = w[n - i - 1];
		w[n - i - 1] = temp;
	}
}

//Doesnt support by microcontroller
/*
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

char* LoadFile(const char* filename)
{
	FILE* textfile = fopen(filename, "r");
	if (textfile == NULL)
		return NULL;

	fseek(textfile, 0L, SEEK_END);
	long numbytes = ftell(textfile);
	fseek(textfile, 0L, SEEK_SET);

	char* text = (char*)calloc(numbytes, sizeof(char));
	if (text == NULL)
		return NULL;

	fread(text, sizeof(char), numbytes, textfile);
	fclose(textfile);

	return text;
}
*/

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
