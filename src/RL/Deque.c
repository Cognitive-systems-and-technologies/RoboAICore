#include "Deque.h"

Deque* createDeque(int max_length, unsigned int size, void (*elementFree) (void* e))
{
	Deque* d = (Deque*)malloc(sizeof(Deque));
	d->elem_size = size;
	d->capacity = max_length;
	d->length = 0;
	d->elemFree = elementFree;
	d->data = (void**)malloc(size * max_length);
	return d;
}

void dequeAppend(Deque* d, void* t)
{
	int id = d->length + 1;
	if (id > d->capacity)
	{
		//delete first
		d->elemFree(d->data[0]);
		//move array
		memmove(&d->data[0], &d->data[1], d->elem_size * d->capacity - 1);
		//set last
		d->data[d->length - 1] = t;
	}
	else
	{
		d->data[id - 1] = t;
		d->length = id;
	}
}

void freeDeque(Deque* d)
{
	for (size_t i = 0; i < d->capacity; i++)
	{
		d->elemFree(d->data[i]);
	}
	free(d);
}