#ifndef DDPG_H
#define DDPG_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdlib.h>

#include "TCommon.h"
#include "Interfaces.h"
#include "Optimizer.h"
#include "Tensor.h"
//#include "ReplayBuffer.h"

#include "Utils.h"
#include "Losses.h"
//#include "geometry/TVec2.h"

typedef struct DDPG
{
	Layer *inpA, *inpC, *inpCA, *actor, *critic;
	Layer *inpAT, *inpCT, *inpCTA, *actor_target, *critic_target;
	Model ActorNet;
	Model CriticNet;

	Model ActorTargetNet;
	Model CriticTargetNet;
	float gamma;
	float tau;
	shape input_shape;
	int num_outputs;
	OptParams par;
}DDPG;

void soft_update(dList* tp, dList*sp, float tau);
DDPG* DDPG_Create(shape state_shape, int n_acts);
Tensor DDPG_Forward(DDPG *brain, Tensor *state);
Tensor DDPG_SelectAction(DDPG* brain, Tensor* state, float eps);

float DDPG_TrainTrace(DDPG* brain, Tensor* states, Tensor* probs, float* rewards, int n, int iter);

void copy_params(Model* dest_m, Model* src_m);
void SoftUpdate_Targets(DDPG* brain);
#ifdef __cplusplus
}
#endif

#endif
