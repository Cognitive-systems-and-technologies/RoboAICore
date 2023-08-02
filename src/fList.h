#ifndef FLOATLIST_H//dynamic list
#define FLOATLIST_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include <stdlib.h>

	typedef struct fList
	{
		int length;
		float* data;
	}fList;

	fList fList_create();
	void fList_realloc(fList* d);//add new elem
	float fList_push(fList* d, float t);//add and assign
	void fList_free(fList* d);//clear list

#ifdef __cplusplus
}
#endif

#endif //!FLOATLIST_H