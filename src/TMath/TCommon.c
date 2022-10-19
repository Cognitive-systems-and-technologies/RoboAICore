#include "TCommon.h"
#include <stdlib.h>
#include <math.h>

float DegToRad(float deg) { return M_PI * deg / 180.0f; }
float RadToDeg(float rad) { rad* (180.0f / M_PI); }
float Lerp(float a, float b, float t) { return a - (a * t) + (b * t); }
float InvLerp(float a, float b, float t) { 	return (t - a) / (b - a); }