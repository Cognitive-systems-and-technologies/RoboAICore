#include "TCommon.h"
#include <stdlib.h>
#include <math.h>

float DegToRad(float deg) { return M_PI * deg / 180.0f; }
float RadToDeg(float rad) { return rad * (180.0f / M_PI); }
float Lerp(float a, float b, float t) { return a - (a * t) + (b * t); }
float InvLerp(float a, float b, float t) { 	return (t - a) / (b - a); }

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