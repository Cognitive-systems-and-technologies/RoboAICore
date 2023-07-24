#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif 
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void WriteToFile(const char* txt, const char* file);
void PrintArray(float* w, int n);
void FillArray(float* w, int n, float v);
#ifdef __cplusplus
}
#endif

#endif
