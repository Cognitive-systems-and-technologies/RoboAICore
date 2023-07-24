#ifndef TARC_H
#define TARC_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdlib.h>
#include <math.h>

typedef struct TArc
{
	float r;
	float startAngle;
	float sweepAngle;
}TArc;

int TArc_IsClockwise(TArc a);
float TArc_Length(TArc a);

#ifdef __cplusplus
}
#endif

#endif
