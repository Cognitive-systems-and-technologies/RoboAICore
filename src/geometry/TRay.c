#include "TRay.h"
#include <stdlib.h>
#include <math.h>

TVec3 TRay_OnRay(TRay r, float dist) 
{
	TVec3 n = TVec3_Norm(r.dir);
	return TVec3_Add(r.org, TVec3_Mul(n, dist));
}