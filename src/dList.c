#include "dList.h"

dList dList_create()
{
	dList l;
	l.data = NULL;
	l.length = 0;
	return l;
}
void dList_realloc(dList* d)
{
	d->length += 1;
	dlElem* tmp = (dlElem*)realloc(d->data, sizeof(dlElem) * d->length);
	if (!tmp) {
		free(d->data);
		d->data = NULL;
		return NULL;
	}
	d->data = tmp;
}
void* dList_push(dList* d, void* t)
{
	dList_realloc(d);
	d->data[d->length - 1].e = t;
	return d->data[d->length - 1].e;
}
void dList_free(dList* d) 
{
	free(d->data);
	d->data = NULL;
}