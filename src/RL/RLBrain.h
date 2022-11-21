#ifndef RLBRAIN_H
#define RLBRAIN_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdlib.h>

#include "TCommon.h"
#include "Interfaces.h"
#include "Optimizer.h"
#include "Sequential.h"
#include "Tensor.h"
#include "ReplayBuffer.h"

typedef struct RLBrain
{
	ReplayBuffer *buffer;
	Net *net;

	shape input_shape;
	int num_outputs;
	OptParams par;
}RLBrain;

RLBrain *RLBrain_Create(shape state_shape, int n_outputs);
Net *RLBrain_CreateNet(shape input_sh, int n_outputs);
void RLBrain_Record(RLBrain *brain, Tensor* state, Tensor* next_state, int action, float reward);
Tensor* RLBrain_Forward(RLBrain *brain, Tensor *state);
float RLBrain_Train(RLBrain *brain);

#ifdef __cplusplus
}
#endif

#endif
