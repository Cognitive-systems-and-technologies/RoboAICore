#ifndef TWEIGHTSI_H
#define TWEIGHTSI_H

#ifdef __cplusplus
extern "C" {
#endif 
#include "TCommon.h"
#include <math.h>

typedef enum RandType {
	R_XAVIER,
	R_XAVIER_NORM,
	R_HE
} RandType;

float xavier_rand(int n);
float xavier_norm_rand(int n, int m);
float he_rand(int n);

#ifdef __cplusplus
}
#endif

#endif