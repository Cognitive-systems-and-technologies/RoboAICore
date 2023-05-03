#ifndef TWEIGHTSI_H
#define TWEIGHTSI_H

#ifdef __cplusplus
extern "C" {
#endif 
#include "TCommon.h"
#include <math.h>

float xavier_rand(int n);
float xavier_norm_rand(int n, int m);
float he_rand(int n);

#ifdef __cplusplus
}
#endif

#endif