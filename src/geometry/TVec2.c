#include "TVec2.h"
#include <stdlib.h>
#include <math.h>

TVec2 TVec2_Create(float x, float y)
{
	TVec2 vec = (TVec2){x, y};
	return vec;
}

TVec2 TVec2_Create2(float v)
{
	TVec2 vec = (TVec2){ v, v };
	return vec;
}

TVec2 TVec2_Mul(TVec2 v, float d) 
{
	return (TVec2) { v.x * d, v.y * d };
}

TVec2 TVec2_Div(TVec2 v, float d)
{
	if (d != 0)
		return (TVec2) { v.x / d, v.y / d};
	else
		return TVec2_Create2(0);
}

TVec2 TVec2_Sub(TVec2 v1, TVec2 v2)
{
	return (TVec2) { v1.x - v2.x, v1.y - v2.y };
}

TVec2 TVec2_Add(TVec2 v1, TVec2 v2)
{
	return (TVec2) { v1.x + v2.x, v1.y + v2.y };
}

TVec2 TVec2_Norm(TVec2 v) 
{
	float le = TVec2_Length(v);
	return TVec2_Div(v, le);
}

float TVec2_Length(TVec2 v) 
{
	return sqrtf(v.x * v.x + v.y * v.y);
}

TVec2 TVec2_Dir(TVec2 org, TVec2 dest) 
{
	return TVec2_Norm(TVec2_Sub(dest, org));
}

float TVec2_Dot(TVec2 v1, TVec2 v2) 
{
	return v1.x * v2.x + v1.y * v2.y;
}

float TVec2_AngleDeg(TVec2 v1, TVec2 v2) 
{
	float sin = v1.x * v2.y - v2.x * v1.y;
	float cos = v1.x * v2.x + v1.y * v2.y;

	return (float)atan2(sin, cos) * (180.f / M_PI);
}