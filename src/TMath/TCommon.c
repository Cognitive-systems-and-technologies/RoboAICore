#include "TCommon.h"
#include <stdlib.h>
#include <math.h>

float DegToRad(float deg) { return M_PI * deg / 180.0f; }
float RadToDeg(float rad) { return rad * (180.0f / M_PI); }
float Lerp(float a, float b, float t) { return a - (a * t) + (b * t); }
float InvLerp(float a, float b, float t) { 	return (t - a) / (b - a); }

float rngFloat() { return  (float)(rand()) / (float)(RAND_MAX); }
int rngInt(int min, int max) { int randNum = rand() % (max - min + 1) + min; return randNum; }

void InsertionSort(float *values, int n) {
	for (size_t i = 1; i < n; ++i) {
		float x = values[i];
		size_t j = i;
		while (j > 0 && values[j - 1] > x) {
			values[j] = values[j - 1];
			--j;
		}
		values[j] = x;
	}
}

float Mean(float *items, int n)
{
	float sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += items[i];
	}
	return sum / n;
}
