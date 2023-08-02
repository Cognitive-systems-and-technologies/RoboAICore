#include "fList.h"

fList fList_create()
{
	fList l;
	l.data = NULL;
	l.length = 0;
	return l;
}
void fList_realloc(fList* d)
{
	d->length += 1;
	float* tmp = (float*)realloc(d->data, sizeof(float) * d->length);
	if (!tmp) {
		free(d->data);
		d->data = NULL;
		return NULL;
	}
	d->data = tmp;
}
float fList_push(fList* d, float t)
{
	fList_realloc(d);
	d->data[d->length - 1] = t;
	return d->data[d->length - 1];
}
void fList_free(fList* d)
{
	free(d->data);
	d->data = NULL;
	d->length = 0;
}