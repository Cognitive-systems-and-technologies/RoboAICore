#ifndef TRAY_H
#define TRAY_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "TVec3.h"

typedef struct TRay
{
	TVec3 org;
	TVec3 dir;
}TRay;

TVec3 TRay_OnRay(TRay r, float dist);

#ifdef __cplusplus
}
#endif

#endif
