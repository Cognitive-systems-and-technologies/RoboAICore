#ifndef RLAGENT_H
#define RLAGENT_H

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdlib.h>

#include "TCommon.h"
#include "Tensor.h"
#include "RLBrain.h"

typedef enum AgentPhase
{
	A_IDLE,
	A_TRAIN,
	A_TEST
}AgentPhase;

typedef struct RLAgent
{
	RLBrain *brain;
	Tensor state;
	float epsilon;
	float decay;
	AgentPhase phase;
}RLAgent;

RLAgent *RLAgent_Create(shape state_shape, int n_outputs);
int RLAgent_Policy(RLAgent *agent, Tensor* s);
int RLAgent_Act(RLAgent *agent, Tensor* s);

#ifdef __cplusplus
}
#endif

#endif
