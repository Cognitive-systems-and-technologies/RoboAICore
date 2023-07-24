#ifndef RBUFFER_H
#define RBUFFER_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdlib.h>
#include <string.h>

#include "Tensor.h"
#include "Interfaces.h"
#include "SimpleDeque.h"
#include "dList.h"

typedef struct
{
	Tensor* state;
	Tensor* next_state;
	int action;
	float reward;
	int done;//bool
}Sample;

typedef struct 
{
	int capacity;
	int batch_size;
	SimpleDeque* buffer;
}ReplayBuffer;

ReplayBuffer *ReplayBuffer_Create(int capacity, int batch_size);
void ReplayBuffer_Record(ReplayBuffer* rBuffer, Tensor* state,
	Tensor* next_state,
	int action,
	float reward, int done);
dList ReplayBuffer_Sample(ReplayBuffer* rb);
void ReplayBuffer_Free(ReplayBuffer *rBuffer);

Sample* createSample(Tensor* state,
	Tensor* next_state,
	int action,
	float reward, int done);
void freeSample(void* sample);

#ifdef __cplusplus
}
#endif

#endif
