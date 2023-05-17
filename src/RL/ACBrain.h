#ifndef ACBRAIN_H
#define ACBRAIN_H

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

typedef struct ACBrain
{
	ReplayBuffer *buffer;
	Net *net;
	float discount;
	shape input_shape;
	int num_outputs;
	OptParams par;
	float critic_loss_weight;
	float actor_loss_weight;
	float entropy_loss_weight;
}ACBrain;

ACBrain*ACBrain_Create(shape state_shape, int n_outputs);
Net *ACBrain_CreateNet(shape input_sh, int n_outputs);
void ACBrain_Record(ACBrain *brain, Tensor* state, Tensor* next_state, int action, float reward, int done);
Tensor* ACBrain_Forward(ACBrain *brain, Tensor *state);
float ACBrain_Train(ACBrain *brain);

#ifdef __cplusplus
}
#endif

#endif
