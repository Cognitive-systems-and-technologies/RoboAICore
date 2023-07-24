#ifndef SIMPLEDEQUE_H
#define SIMPLEDEQUE_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

	typedef struct DequeElem
	{
		void* elem;
	}DequeElem;

	typedef struct SimpleDeque
	{
		int capacity;
		int length;
		DequeElem* data;
	}SimpleDeque;

	SimpleDeque* createDeque(int capacity);

	void dequeAppend(SimpleDeque* d, DequeElem t, void (*elementFree) (void* e));

	void freeDeque(SimpleDeque* d, void (*elementFree) (void* e));

#ifdef __cplusplus
}
#endif

#endif // !SIMPLEDEQUE_H