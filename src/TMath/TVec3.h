#ifndef TVEC3_H
#define TVEC3_H

#ifdef __cplusplus
extern "C" {
#endif 

typedef struct TVec3
{
	float x;
	float y;
	float z;
}TVec3;

TVec3 TVec3_Create(float x, float y, float z);
TVec3 TVec3_Create2(float all);

TVec3 TVec3_Mul(TVec3 v, float d);
TVec3 TVec3_Div(TVec3 v, float d);
TVec3 TVec3_Sub(TVec3 v1, TVec3 v2);
TVec3 TVec3_Norm(TVec3 v);
TVec3 TVec3_Cross(TVec3 v1, TVec3 v2);
TVec3 TVec3_Dir(TVec3 org, TVec3 dest); //direction vector
float TVec3_Length(TVec3 v);
float TVec3_Dot(TVec3 v1, TVec3 v2);
float TVec3_AngleRad(TVec3 v1, TVec3 v2);

#ifdef __cplusplus
}
#endif

#endif
