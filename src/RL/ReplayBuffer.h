#ifndef RBUFFER_H
#define RBUFFER_H

#ifdef __cplusplus
extern "C" {
#endif 

#include "Tensor.h"
#include "Interfaces.h"
typedef struct
{
	Tensor state;
	float action;
	float reward;
	Tensor next_state;
}Sample;
typedef struct 
{
	int counter;
	int capacity;
	int batch_size;
	Sample* buffer;
}ReplayBuffer;

ReplayBuffer *ReplayBuffer_Create(int capacity, int batch_size);
void ReplayBuffer_Record(ReplayBuffer* rb, Sample *s);
Sample ReplayBuffer_Sample(ReplayBuffer* rb);

#ifdef __cplusplus
}
#endif

#endif
