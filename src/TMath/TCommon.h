#ifndef TCOMMON_H
#define TCOMMON_H

#ifdef __cplusplus
extern "C" {
#endif 

# define M_PI 3.14159265358979323846 //pi 

float DegToRad(float deg);
float RadToDeg(float rad);
float Lerp(float a, float b, float t);
float InvLerp(float a, float b, float t);

#ifdef __cplusplus
}
#endif

#endif
