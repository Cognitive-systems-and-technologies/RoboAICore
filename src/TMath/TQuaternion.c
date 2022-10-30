#include "TQuaternion.h"
#include <stdlib.h>
#include <math.h>

TQuaternion TQuaternion_Create(float x, float y, float z, float w)
{
	TQuaternion vec = (TQuaternion){x, y, z, w};
	return vec;
}

TQuaternion TQuaternion_CreateV(TVec3 v, float w) 
{
	return (TQuaternion){v.x, v.y, v.z, w};
}

TQuaternion TQuaternion_FromVec3(TVec3 axis, float angleRadian) 
{
    TQuaternion q;
    float m = TVec3_Length(axis);
    if (m > 0.0001f)
    {
        float ca = cosf(angleRadian * 0.5f);
        float sa = sinf(angleRadian * 0.5f);
        q.x = axis.x / m * sa;
        q.y = axis.y / m * sa;
        q.z = axis.z / m * sa;
        q.w = ca;
    }
    else
    {
        q.w = 1; q.x = 0; q.y = 0; q.z = 0;
    }
    return q;
}

TQuaternion TQuaternion_Norm(TQuaternion v) 
{
	TQuaternion r = v;
    float m = v.w * v.w + v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0.0001f)
    {
        m = sqrtf(m);
        r.w /= m;
        r.x /= m;
        r.y /= m;
        r.z /= m;
    }
    else
    {
        r.w = 1.f; r.x = 0; r.y = 0; r.z = 0;
    }
	return r;
}

TQuaternion TQuaternion_Conjugate(TQuaternion v) 
{
    return (TQuaternion){-v.x, -v.y, -v.z, v.w};
}

TQuaternion TQuaternion_Mul(TQuaternion q1, TQuaternion q2) 
{
    float nw = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    float nx = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    float ny = q1.w * q2.y + q1.y * q2.w + q1.z * q2.x - q1.x * q2.z;
    float nz = q1.w * q2.z + q1.z * q2.w + q1.x * q2.y - q1.y * q2.x;
    return (TQuaternion) { nx, ny, nz, nw };
}

TQuaternion TQuaternion_Euler(float x, float y, float z) 
{
    float c1 = cosf(x);
    float s1 = sinf(x);
    float c2 = cosf(y);
    float s2 = sinf(y);
    float c3 = cosf(z);
    float s3 = sinf(z);
    float wn = sqrtf(1.0f + c1 * c2 + c1 * c3 - s1 * s2 * s3 + c2 * c3) / 2.0f;
    float w4 = (4.0f * wn);
    float xn = (c2 * s3 + c1 * s3 + s1 * s2 * c3) / w4;
    float yn = (s1 * c2 + s1 * c3 + c1 * s2 * s3) / w4;
    float zn = (-s1 * s3 + c1 * s2 * c3 + s2) / w4;

    return (TQuaternion) { xn, yn, zn, wn };
}

TVec3 TQuaternion_Rotate(TQuaternion q, TVec3 pt)
{
    //q = TQuaternion_Norm(q);
    TQuaternion q1 = TQuaternion_Norm(q);
    q1 = TQuaternion_Conjugate(q1);

    TQuaternion qNode = (TQuaternion){ pt.x, pt.y, pt.z, 0 };
    qNode = TQuaternion_Mul(TQuaternion_Mul(q, qNode), q1);

    pt.x = qNode.x;
    pt.y = qNode.y;
    pt.z = qNode.z;

    return pt;
}