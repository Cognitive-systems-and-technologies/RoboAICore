#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Optimizer.h"
#include <string.h>

typedef struct Wrapper { void* elem; }Wrapper;

void shuffle(TPair* array, int n);
//#define shuffle_type(type, array, n) ({for (int i = 0; i < n - 1; i++){size_t j = i + rand() / (RAND_MAX / (n - i) + 1); type t = array[j]; array[j] = array[i]; array[i] = t;}})
#ifdef __cplusplus
}
#endif

#endif
