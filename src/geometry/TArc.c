#include "TArc.h"

int TArc_IsClockwise(TArc a) 
{
	return a.sweepAngle > 0 ? 1 : 0;
}

float TArc_Length(TArc a) 
{
	return a.r * fabs(a.sweepAngle);
}