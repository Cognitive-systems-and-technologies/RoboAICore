#ifndef TCOMMON_H
#define TCOMMON_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdlib.h>
# define M_PI 3.14159265358979323846f //pi 

float DegToRad(float deg);
float RadToDeg(float rad);
float Lerp(float a, float b, float t);
float InvLerp(float a, float b, float t);

float rngFloat();
int rngInt(int min, int max);
float rngNormal();
int find_ceil(int* arr, int r, int l, int h);
int rng_by_prob(float* prob, int n);

void InsertionSort(float* values, int n);
float Mean(float* items, int n);
float Derivative(float (*f)(float), float x0);
#ifdef __cplusplus
}
#endif

#endif
