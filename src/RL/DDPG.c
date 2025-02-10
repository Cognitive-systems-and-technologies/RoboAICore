#include "DDPG.h"

DDPG* DDPG_Create(shape state_shape, int n_acts)
{
	DDPG* brain = malloc(sizeof(DDPG));
	if (!brain)
	{
		return NULL;
	}
	brain->input_shape = (shape){ state_shape.w, state_shape.h, state_shape.d };
	brain->num_outputs = n_acts;
	brain->gamma = 0.99f;
	//================== main networks ====================
	brain->ActorNet = Model_Create();
	brain->inpA = Model_AddLayer(&brain->ActorNet, Input_Create(brain->input_shape));
	Layer* l = Model_AddLayer(&brain->ActorNet, Dense_Create(64, R_XAVIER, brain->inpA));
	l = Model_AddLayer(&brain->ActorNet, Dense_Create(64, R_XAVIER, l));
	l = Model_AddLayer(&brain->ActorNet, Dense_Create(n_acts, R_XAVIER, l));
	brain->actor = Model_AddLayer(&brain->ActorNet, TanhA_Create(l));

	brain->CriticNet = Model_Create();
	brain->inpC = Model_AddLayer(&brain->CriticNet, Input_Create(brain->input_shape));//observations
	brain->inpCA = Model_AddLayer(&brain->CriticNet, Input_Create((shape){1,1,n_acts}));//actions
	Layer* l2 = Model_AddLayer(&brain->CriticNet, Dense_Create(64, R_XAVIER, brain->inpC));//obs
	l2 = Model_AddLayer(&brain->CriticNet, Conc_Create(l2, brain->inpCA));//obs+acts
	l2 = Model_AddLayer(&brain->CriticNet, Dense_Create(64, R_XAVIER, l2));
	brain->critic = Model_AddLayer(&brain->CriticNet, Dense_Create(1, R_XAVIER, l2));

	//================== target networks ====================
	brain->ActorTargetNet = Model_Create();
	brain->inpAT = Model_AddLayer(&brain->ActorTargetNet, Input_Create(brain->input_shape));
	Layer* lt = Model_AddLayer(&brain->ActorTargetNet, Dense_Create(64, R_XAVIER, brain->inpAT));
	lt = Model_AddLayer(&brain->ActorTargetNet, Dense_Create(64, R_XAVIER, lt));
	lt = Model_AddLayer(&brain->ActorTargetNet, Dense_Create(n_acts, R_XAVIER, lt));
	brain->actor_target = Model_AddLayer(&brain->ActorTargetNet, TanhA_Create(lt));

	brain->CriticTargetNet = Model_Create();
	brain->inpCT = Model_AddLayer(&brain->CriticTargetNet, Input_Create(brain->input_shape));//observations
	brain->inpCTA = Model_AddLayer(&brain->CriticTargetNet, Input_Create((shape) { 1, 1, n_acts}));//actions
	Layer* l2t = Model_AddLayer(&brain->CriticTargetNet, Dense_Create(64, R_XAVIER, brain->inpCT));//obs
	l2t = Model_AddLayer(&brain->CriticTargetNet, Conc_Create(l2t, brain->inpCTA));//obs+acts
	l2t = Model_AddLayer(&brain->CriticTargetNet, Dense_Create(64, R_XAVIER, l2t));
	brain->critic_target = Model_AddLayer(&brain->CriticTargetNet, Dense_Create(1, R_XAVIER, l2t));

	brain->par = OptParams_Create();
	brain->par.method = ADAN;
	brain->par.learning_rate = 0.00001f;

	brain->tau = 0.01f;
	//copy parameters from mains to targets
	copy_params(&brain->ActorTargetNet, &brain->ActorNet);
	copy_params(&brain->CriticTargetNet, &brain->CriticNet);
	return brain;
}

void copy_params(Model *dest_m, Model *src_m)
{
	dList dst = Model_getGradients(dest_m);
	dList src = Model_getGradients(src_m);
	for (int i = 0; i < src.length; i++)
	{
		Tensor* target = (Tensor*)dst.data[i].e;
		Tensor* source = (Tensor*)src.data[i].e;
		Tensor_CopyData(target, source);
	}
	dList_free(&dst);
	dList_free(&src);
}

void soft_update(dList* tp, dList* sp, float tau) 
{
	for (int i = 0; i < tp->length; i++)
	{
		Tensor* target = (Tensor*)tp->data[i].e;
		Tensor* source = (Tensor*)sp->data[i].e;
		for (int j = 0; j < target->n; j++)
		{
			target->w[j] = (1.f - tau) * target->w[j] + tau * source->w[j];
		}
	}
}

Tensor DDPG_Forward(DDPG* brain, Tensor* state)
{
	brain->inpA->input = state;
	Model_Forward(&brain->ActorNet);
	Tensor prob = Tensor_CreateCopy(&brain->actor->output);
	return prob;
}

Tensor DDPG_SelectAction(DDPG* brain, Tensor* state, float eps)
{
	brain->inpA->input = state;
	Model_Forward(&brain->ActorNet);
	Tensor prob = Tensor_CreateCopy(&brain->actor->output);
	for (size_t i = 0; i < prob.n; i++)
	{
		float res = prob.w[i] + rngNormal()*eps;
		res = Clamp(res, -1.f, 1.f);
		prob.w[i] = res;
	}
	return prob;
}

float DDPG_TrainTrace(DDPG* brain, Tensor* states, Tensor* probs, float* rewards, int n, int iter)
{
	float q_total_loss = 0;
	for (int i = 0; i < n-1; i++)
	{
		//get actor target actions
		brain->inpAT->input = &states[i + 1];
		Model_Forward(&brain->ActorTargetNet);
		Tensor next_action = Tensor_CreateCopy(&brain->actor_target->output);

		brain->inpCT->input = &states[i+1];
		brain->inpCTA->input = &next_action;
		Model_Forward(&brain->CriticTargetNet);
		Tensor target_Q = Tensor_CreateCopy(&brain->critic_target->output);
		target_Q.w[0] = target_Q.w[0] * brain->gamma + rewards[i];

		brain->inpC->input = &states[i];
		brain->inpCA->input = &probs[i];
		Model_Forward(&brain->CriticNet);
		Tensor* current_Q = &brain->critic->output;

		float q_loss = MSE_Loss(current_Q, &target_Q);
		q_total_loss += q_loss;
		Model_Backward(&brain->CriticNet);
		OptimizeModel(&brain->CriticNet, &brain->par);

		//setup current actor
		brain->inpA->input = &states[i];
		Model_Forward(&brain->ActorNet);
		Tensor new_actions = Tensor_CreateCopy(&brain->actor->output);

		brain->inpC->input = &states[i];
		brain->inpCA->input = &new_actions;
		Model_Forward(&brain->CriticNet);
		Tensor* new_Q = &brain->critic->output;

		FillArray(new_Q->dw, new_Q->n, 1.f);
		Model_Backward(&brain->CriticNet);
		for (size_t j = 0; j < brain->actor->output.n; j++)
		{
			brain->actor->output.dw[j] = brain->inpCA->output.dw[j];
		}
		Model_CLearGrads(&brain->CriticNet);

		Model_Backward(&brain->ActorNet);
		OptimizeModel(&brain->ActorNet, &brain->par);

		Tensor_Free(&next_action);
		Tensor_Free(&new_actions);
		Tensor_Free(&target_Q);
	}
	q_total_loss /= (float)n;
	printf("q loss: %f\n", q_total_loss);

	if (iter % 100 == 0) {
		printf("====================Soft update targets=====================\n");
		SoftUpdate_Targets(brain);
	}
	return -1.f;
}

void SoftUpdate_Targets(DDPG* brain) 
{
	printf("Soft update targets\n");
	//get grads
	dList ac = Model_getGradients(&brain->ActorNet);
	dList acT = Model_getGradients(&brain->ActorTargetNet);

	dList cr = Model_getGradients(&brain->CriticNet);
	dList crT = Model_getGradients(&brain->CriticTargetNet);

	soft_update(&acT, &ac, brain->tau);
	soft_update(&crT, &cr, brain->tau);

	dList_free(&ac);
	dList_free(&acT);
	dList_free(&cr);
	dList_free(&crT);
}
