#ifndef TD3_H
#define TD3_H

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

typedef struct TD3
{
	Layer *inpA, *inpC, *inpCA, *actor, *critic, *Q1, *Q2;
	Layer *inpAT, *inpCT, *inpCTA, *actor_target, *critic_target, *QT1, *QT2;
	Model ActorNet;
	Model CriticNet;

	Model ActorTargetNet;
	Model CriticTargetNet;
	float gamma;
	float tau;
	shape input_shape;
	int num_outputs;
	float noise_clip;
	int update_frq;
	OptParams par;
	OrnsteinUhlenbeckNoise noise;
}TD3;

void soft_update(dList* tp, dList*sp, float tau);
TD3* TD3_Create(shape state_shape, int n_acts);
Tensor TD3_Forward(TD3 *brain, Tensor *state);
Tensor TD3_SelectAction(TD3* brain, Tensor* state, float eps);

float TD3_TrainTrace(TD3* brain, Tensor* states, Tensor* last_state, Tensor* probs, float* rewards, int n, int iter);

void copy_params(Model* dest_m, Model* src_m);
void SoftUpdate_Targets(TD3* brain);
#ifdef __cplusplus
}
#endif

#endif
