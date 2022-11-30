#include "ValueDeque.h"

ValueDeque* createValueDeque(int capacity)
{
	ValueDeque* d = (ValueDeque*)malloc(sizeof(ValueDeque));
	if (!d)
	{
		printf("Deque allocation error!");
		return NULL;
	}
	d->capacity = capacity;
	d->length = 0;
	d->data = (float*)malloc(sizeof(float) * capacity);
	if (!d->data)
	{
		printf("Deque allocation error!");
		free(d);
		return NULL;
	}
	for (size_t i = 0; i < capacity; i++)
	{
		d->data[i] = 0;
	}
	return d;
}

void ValueDequeAppend(ValueDeque* d, float t)
{
	int id = d->length + 1;
	if (id > d->capacity)
	{
		//move array
		memmove(&d->data[0], &d->data[1], sizeof(float) * d->capacity - 1);
		//set last
		d->data[d->length - 1] = t;
	}
	else
	{
		d->data[id - 1] = t;
		d->length = id;
	}
}

void freeValueDeque(ValueDeque* d)
{
	free(d->data);
	free(d);
}