#include "RLAgent.h"

RLAgent *RLAgent_Create(shape state_shape, int n_outputs)
{
	RLAgent *agent = malloc(sizeof(RLAgent));
	if(!agent)
	{
		return NULL;
	}
    agent->state = Tensor_Create(state_shape, 0);
	agent->brain = RLBrain_Create(state_shape, n_outputs);
	agent->epsilon = 0.9f;
	agent->phase = A_TRAIN;
	return agent;
}

int RLAgent_Policy(RLAgent *agent, Tensor* s)
{
	Tensor* y = RLBrain_Forward(agent->brain, s);
	shape max = T_Argmax(y);
	int act = max.d;
	return act;
}

int RLAgent_Act(RLAgent *agent, Tensor* s)
{
	if (rngFloat() <= agent->epsilon) {
		int ra = rngInt(0, agent->brain->num_outputs-1);
		return ra;
	}
	else {
		int act = RLAgent_Policy(agent, s);
		return act;
	}
}
