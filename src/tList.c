#include "tList.h"

tList tList_create()
{
	tList l;
	l.data = NULL;
	l.length = 0;
	return l;
}
void tList_realloc(tList* d)
{
	d->length += 1;
	Tensor* tmp = (Tensor*)realloc(d->data, sizeof(Tensor) * d->length);
	if (!tmp) {
		free(d->data);
		d->data = NULL;
		return NULL;
	}
	d->data = tmp;
}
Tensor tList_push(tList* d, Tensor* t)
{
	tList_realloc(d);
	d->data[d->length - 1] = Tensor_CreateCopy(t);
	return d->data[d->length - 1];
}
void tList_free(tList* d)
{
	for (size_t i = 0; i < d->length; i++)
	{
		Tensor_Free(&d->data[i]);
	}
	free(d->data);
	d->data = NULL;
	d->length = 0;
}