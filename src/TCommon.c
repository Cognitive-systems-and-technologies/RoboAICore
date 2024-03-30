#include "TCommon.h"
#include <stdlib.h>
#include <math.h>

float DegToRad(float deg) { return M_PI * deg / 180.0f; }
float RadToDeg(float rad) { return rad * (180.0f / M_PI); }
float Lerp(float a, float b, float t) { return a - (a * t) + (b * t); }
float InvLerp(float a, float b, float t) { 	return (t - a) / (b - a); }

void InsertionSort(float* values, int n) {
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

float Mean(float* items, int n)
{
	float sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += items[i];
	}
	return sum / n;
}

float rngFloat() { return  (float)(rand()) / (float)(RAND_MAX); }
int rngInt(int min, int max) { int randNum = rand() % (max - min + 1) + min; return randNum; }
float rngNormal() {
    float u = ((float)rand() / (RAND_MAX)) * 2.f - 1.f;
    float v = ((float)rand() / (RAND_MAX)) * 2.f - 1.f;
    float r = u * u + v * v;

    while (r == 0 || r > 1) 
    {
        u = ((float)rand() / (RAND_MAX)) * 2.f - 1.f;
        v = ((float)rand() / (RAND_MAX)) * 2.f - 1.f;
        r = u * u + v * v;
    }
    
    float c = sqrtf(-2.f * logf(r) / r);
    return u * c;
}

//from:
//https://www.geeksforgeeks.org/random-number-generator-in-arbitrary-probability-distribution-fashion/
int find_ceil(int* arr, int r, int l, int h)
{
    int mid = 0;
    while (l < h)
    {
        mid = l + ((h - l) >> 1);
        (r > arr[mid]) ? (l = mid + 1) : (h = mid);
    }
    return (arr[l] >= r) ? l : -1;
}

int rng_by_prob(float* prob, int n)
{
    int* prefix = createIntArray(n);
    if (prefix != NULL)
    {
        prefix[0] = (int)(prob[0] * 100.f);
        for (int i = 1; i < n; ++i)
            prefix[i] = prefix[i - 1] + (int)(prob[i] * 100.f);
        int r = (rand() % prefix[n - 1]) + 1;
        int indexc = find_ceil(prefix, r, 0, n - 1);
        free(prefix);
        return indexc;
    }
    return 0;
}

float Derivative(float (*f)(float), float x0)
{
    const float delta = 1.0e-6; //small offset
    float x1 = x0 - delta;
    float x2 = x0 + delta;
    float y1 = f(x1);
    float y2 = f(x2);
    return (y2 - y1) / (x2 - x1);
}
