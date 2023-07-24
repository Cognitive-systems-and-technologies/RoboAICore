#ifndef TVEC2_H
#define TVEC2_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "TCommon.h"

typedef struct TVec2
{
	float x;
	float y;
}TVec2;

TVec2 TVec2_Create(float x, float y);
TVec2 TVec2_Create2(float all);

TVec2 TVec2_Mul(TVec2 v, float d);
TVec2 TVec2_Div(TVec2 v, float d);
TVec2 TVec2_Sub(TVec2 v1, TVec2 v2);
TVec2 TVec2_Add(TVec2 v1, TVec2 v2);
TVec2 TVec2_Norm(TVec2 v);

TVec2 TVec2_Dir(TVec2 org, TVec2 dest); //direction vector

float TVec2_Length(TVec2 v);

float TVec2_Dot(TVec2 v1, TVec2 v2);
float TVec2_AngleDeg(TVec2 v1, TVec2 v2);

#ifdef __cplusplus
}
#endif

#endif
