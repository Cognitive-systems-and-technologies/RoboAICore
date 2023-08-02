#ifndef TENSORLIST_H//dynamic list
#define TENSORLIST_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include <stdlib.h>

	typedef struct tList
	{
		int length;
		Tensor* data;
	}tList;

	tList tList_create();
	void tList_realloc(tList* d);//add new elem
	Tensor tList_push(tList* d, Tensor* t);//add and assign
	void tList_free(tList* d);//clear list

#ifdef __cplusplus
}
#endif

#endif //!TENSORLIST_H