#include "ReplayBuffer.h"

ReplayBuffer *ReplayBuffer_Create(int capacity, int batch_size)
{
	ReplayBuffer *rb = malloc(sizeof(ReplayBuffer));
    if (!rb) 
    {
        printf("Replay buffer allocation error");
        return NULL;
    }
	rb->capacity = capacity;
	rb->batch_size = batch_size;
    rb->buffer = createDeque(capacity);
	return rb;
}

void ReplayBuffer_Record(ReplayBuffer *rBuffer, Tensor* state,
	Tensor* next_state,
	int action,
	float reward, int done)
{
    Sample* s = createSample(state, next_state, action, reward, done);
    dequeAppend(rBuffer->buffer, (DequeElem){ s }, freeSample);
}

dList ReplayBuffer_Sample(ReplayBuffer* rb)
{
    dList lst = dList_create();
    //sample buffer
    return lst;
}

Sample* createSample(Tensor* state,
    Tensor* next_state,
    int action,
    float reward, int done)
{
    Sample* s = malloc(sizeof(Sample));
    if (!s)
    {
        printf("Sample allocation error!");
        return NULL;
    }
    s->action = action;
    s->reward = reward;
    //s->state = Tensor_CreateCopy(state);
    //s->next_state = Tensor_CreateCopy(next_state);
    s->done = done;
    return s;
}

void freeSample(void* sample)
{
	Sample* s = (Sample*)sample;
	Tensor_Free(s->state);
	Tensor_Free(s->next_state);
	free(s);
}

void ReplayBuffer_Free(ReplayBuffer* rBuffer) 
{
    freeDeque(rBuffer->buffer, freeSample);
    free(rBuffer);
}