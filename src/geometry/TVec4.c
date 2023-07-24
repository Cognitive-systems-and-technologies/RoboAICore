#include "TVec4.h"
#include <stdlib.h>
#include <math.h>

TVec4 TVec4_Create(float x, float y, float z, float w)
{
	TVec4 vec = (TVec4){x, y, z, w};
	return vec;
}

TVec4 TVec4_Create3(float x, float y, float z)
{
	TVec4 vec = (TVec4){ x, y, z, 1 };
	return vec;
}

TVec4 TVec4_Create1(float v)
{
	TVec4 vec = (TVec4){ v, v, v, v };
	return vec;
}

TVec4 TVec4_Mul(TVec4 v, float d) 
{
	return (TVec4) { v.x * d, v.y * d, v.z * d, v.w * d };
}

TVec4 TVec4_Div(TVec4 v, float d)
{
	if (d != 0)
		return (TVec4) { v.x / d, v.y / d, v.z / d, v.w / d };
	else
		return TVec4_Create1(0);
}

TVec4 TVec4_Sub(TVec4 v1, TVec4 v2)
{
	return (TVec4) { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w };
}

TVec4 TVec4_Norm(TVec4 v) 
{
	TVec4 r = v;
	float norm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
	float invNorm = 1.0f / norm;

	r.x *= invNorm;
	r.y *= invNorm;
	r.z *= invNorm;
	r.w *= invNorm;
	return r;
}

float TVec4_Dot(TVec4 v1, TVec4 v2) 
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}