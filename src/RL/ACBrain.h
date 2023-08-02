#ifndef ACBRAIN_H
#define ACBRAIN_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdlib.h>

#include "TCommon.h"
#include "Interfaces.h"
#include "Optimizer.h"
#include "Tensor.h"
#include "ReplayBuffer.h"

#include "Utils.h"
#include "Losses.h"

typedef struct ACBrain
{
	Layer *inpA, *inpC, *actor, *critic;
	Model ActorNet;
	Model CriticNet;
	float gamma;
	float I;
	float discount;
	shape input_shape;
	int num_outputs;
	OptParams par;
}ACBrain;

ACBrain*ACBrain_Create(shape state_shape, int n_outputs);
//Model ACBrain_CreateNet(shape input_sh, int n_outputs);
Tensor ACBrain_Forward(ACBrain *brain, Tensor *state);
float ACBrain_TrainTrace(ACBrain* brain, Tensor* states, float* rewards, float* actions, int n);

#ifdef __cplusplus
}
#endif

#endif
