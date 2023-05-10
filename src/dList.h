#ifndef DLIST_H//dynamic list
#define DLIST_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdlib.h>
#define getLElem(E, L, I)  ((E*)L.data[I].e)

	typedef struct dlElem
	{
		void* e;
	}dlElem;

	typedef struct dList
	{
		int length;
		dlElem* data;
	}dList;

	dList dList_create();
	void dList_realloc(dList* d);//add new elem
	void dList_push(dList* d, void* t);//add and assign
	void dList_free(dList* d);//clear list

#ifdef __cplusplus
}
#endif

#endif //!DLIST_H