#ifndef TVEC4_H
#define TVEC4_H

#ifdef __cplusplus
extern "C" {
#endif 

typedef struct TVec4
{
	float x;
	float y;
	float z;
	float w;
}TVec4;

TVec4 TVec4_Create(float x, float y, float z, float w);
TVec4 TVec4_Create1(float x, float y, float z);
TVec4 TVec4_Create2(float all);

TVec4 TVec4_Mul(TVec4 v, float d);
TVec4 TVec4_Div(TVec4 v, float d);
TVec4 TVec4_Sub(TVec4 v1, TVec4 v2);
TVec4 TVec4_Norm(TVec4 v);
float TVec4_Dot(TVec4 v1, TVec4 v2);

#ifdef __cplusplus
}
#endif

#endif
