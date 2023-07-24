#include "TVec3.h"
#include <stdlib.h>
#include <math.h>

TVec3 TVec3_Create(float x, float y, float z)
{
	TVec3 vec = (TVec3){x, y, z};
	return vec;
}

TVec3 TVec3_Create2(float v)
{
	TVec3 vec = (TVec3){ v, v, v };
	return vec;
}

TVec3 TVec3_Mul(TVec3 v, float d) 
{
	return (TVec3) { v.x * d, v.y * d, v.z * d };
}

TVec3 TVec3_Div(TVec3 v, float d)
{
	if (d != 0)
		return (TVec3) { v.x / d, v.y / d, v.z / d };
	else
		return TVec3_Create2(0);
}

TVec3 TVec3_Sub(TVec3 v1, TVec3 v2)
{
	return (TVec3) { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
}

TVec3 TVec3_Add(TVec3 v1, TVec3 v2)
{
	return (TVec3) { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
}

TVec3 TVec3_Norm(TVec3 v) 
{
	return TVec3_Div(v, TVec3_Length(v));
}

float TVec3_Length(TVec3 v) 
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

TVec3 TVec3_Cross(TVec3 v1, TVec3 v2) 
{
	return (TVec3) {
		v1.y* v2.z - v1.z * v2.y,
		v1.z* v2.x - v1.x * v2.z,
		v1.x* v2.y - v1.y * v2.x
	};
}

float TVec3_Dot(TVec3 v1, TVec3 v2) 
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

TVec3 TVec3_Dir(TVec3 org, TVec3 dest) 
{
	return TVec3_Norm(TVec3_Sub(dest, org));
}

float TVec3_AngleRad(TVec3 v1, TVec3 v2) 
{
	float l1 = TVec3_Length(v1);
	float l2 = TVec3_Length(v2);
	float dot = TVec3_Dot(TVec3_Div(v1, l1), TVec3_Div(v2, l2));
	return acosf(dot);
}

TVec3 TVec3_Middle(TVec3 org, TVec3 dest) 
{
	TVec3 v = TVec3_Sub(dest, org);
	float l = TVec3_Length(v) * 0.5f;
	TVec3 n = TVec3_Norm(v);
	return TVec3_Add(org, TVec3_Mul(n, l));

}