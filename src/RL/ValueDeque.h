#ifndef VALUEDEQUE_H
#define VALUEDEQUE_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ValueDeque
{
	int capacity;
	int length;
	float* data;
}ValueDeque;

ValueDeque* createValueDeque(int capacity);

void ValueDequeAppend(ValueDeque* d, float t);

void freeValueDeque(ValueDeque* d);

#ifdef __cplusplus
}
#endif

#endif // !VALUEDEQUE_H

