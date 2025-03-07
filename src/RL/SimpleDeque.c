#include "SimpleDeque.h"

SimpleDeque* createDeque(int capacity)
{
	SimpleDeque* d = (SimpleDeque*)malloc(sizeof(SimpleDeque));
	if (!d)
	{
		printf("Deque allocation error!");
		return NULL;
	}
	d->capacity = capacity;
	d->length = 0;
	d->data = (DequeElem*)malloc(sizeof(DequeElem) * capacity);
	if (!d->data)
	{
		printf("Deque data allocation error!");
		free(d);
		return NULL;
	}
	return d;
}

void dequeAppend(SimpleDeque* d, void *t, void (*elementFree) (void* e))
{
	int id = d->length + 1;
	if (id > d->capacity)
	{
		//delete first
		elementFree(d->data[0].elem);
		free(d->data[0].elem);
		d->data[0].elem = NULL;
		//move array
		memmove(&d->data[0], &d->data[1], sizeof(DequeElem) * d->capacity - 1);
		//set last
		d->data[d->length - 1].elem = t;
	}
	else
	{
		d->data[id - 1].elem = t;
		d->length = id;
	}
}

void freeDeque(SimpleDeque* d, void (*elementFree) (void* e))
{
	for (int i = 0; i < d->length; i++)
	{
		elementFree(d->data[i].elem);
		free(d->data[i].elem);
	}
	free(d->data);
	free(d);
}