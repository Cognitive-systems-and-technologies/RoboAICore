#ifndef DEQUE_H
#define DEQUE_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <string.h>

typedef struct
{
	int capacity;
	volatile int length;
	unsigned int elem_size;
	void** data;
	void (*elemFree) (void *e);
}Deque;

Deque* createDeque(int max_length, unsigned int size, void (*elementFree) (void* e));

void dequeAppend(Deque* d, void* t);

void freeDeque(Deque* d);

#ifdef __cplusplus
}
#endif

#endif // !DEQUE_H

