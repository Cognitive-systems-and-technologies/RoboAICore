#ifndef RLBRAIN_H
#define RLBRAIN_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdlib.h>

#include "TCommon.h"
#include "Interfaces.h"
#include "Optimizer.h"
#include "Tensor.h"
#include "ReplayBuffer.h"
#include "Losses.h"

typedef struct RLBrain
{
	Layer* inp, *out;
	ReplayBuffer *buffer;
	Model net;
	float discount;
	shape input_shape;
	int num_outputs;
	OptParams par;
}RLBrain;

RLBrain *RLBrain_Create(shape state_shape, int n_outputs);
//Model RLBrain_CreateNet(shape input_sh, int n_outputs);
void RLBrain_Record(RLBrain *brain, Tensor* state, Tensor* next_state, int action, float reward, int done);
Tensor RLBrain_Forward(RLBrain *brain, Tensor *state);
float RLBrain_Train(RLBrain *brain);
float RLBrain_TrainTrace(RLBrain* brain, Tensor* states, float* rewards, float* actions, int n);

#ifdef __cplusplus
}
#endif

#endif
