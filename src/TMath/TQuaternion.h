#ifndef TQUATERNION_H
#define TQUATERNION_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "TVec4.h"
#include "TVec3.h"

typedef struct TQuaternion
{
	float x;
	float y;
	float z;
	float w;
}TQuaternion;

TQuaternion TQuaternion_Create(float x, float y, float z, float w);
TQuaternion TQuaternion_CreateV(TVec3 v, float w);
TQuaternion TQuaternion_FromVec3(TVec3 axis, float angleRadian);
TQuaternion TQuaternion_Norm(TQuaternion v);
TQuaternion TQuaternion_Conjugate(TQuaternion v);
TQuaternion TQuaternion_Mul(TQuaternion q1, TQuaternion q2);
TQuaternion TQuaternion_Euler(float x, float y, float z);

TVec3 TQuaternion_Rotate(TQuaternion q, TVec3 pt);
#ifdef __cplusplus
}
#endif

#endif
