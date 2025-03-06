#include "TD3.h"

TD3* TD3_Create(shape state_shape, int n_acts)
{
	TD3* brain = malloc(sizeof(TD3));
	if (!brain)
	{
		return NULL;
	}
	brain->input_shape = (shape){ state_shape.w, state_shape.h, state_shape.d };
	brain->num_outputs = n_acts;
	brain->gamma = 0.99f;
	brain->noise_clip = 0.5f;
	brain->update_frq = 100;
	//================== main networks ====================
	brain->ActorNet = Model_Create();
	brain->inpA = Model_AddLayer(&brain->ActorNet, Input_Create(brain->input_shape));
	Layer* l = Model_AddLayer(&brain->ActorNet, Dense_Create(64, R_XAVIER, brain->inpA));
	l = Model_AddLayer(&brain->ActorNet, Relu_Create(l));
	l = Model_AddLayer(&brain->ActorNet, Dense_Create(64, R_XAVIER, l));
	l = Model_AddLayer(&brain->ActorNet, Relu_Create(l));
	l = Model_AddLayer(&brain->ActorNet, Dense_Create(n_acts, R_XAVIER, l));
	brain->actor = Model_AddLayer(&brain->ActorNet, TanhA_Create(l));

	brain->CriticNet = Model_Create();
	brain->inpC = Model_AddLayer(&brain->CriticNet, Input_Create(brain->input_shape));//observations
	brain->inpCA = Model_AddLayer(&brain->CriticNet, Input_Create((shape){1,1,n_acts}));//actions
	Layer* conc1 = Model_AddLayer(&brain->CriticNet, Conc_Create(brain->inpC, brain->inpCA));//obs+acts

	Layer* l1 = Model_AddLayer(&brain->CriticNet, Dense_Create(64, R_XAVIER, conc1));
	l1 = Model_AddLayer(&brain->CriticNet, Relu_Create(l1));
	l1 = Model_AddLayer(&brain->CriticNet, Dense_Create(64, R_XAVIER, l1));
	l1 = Model_AddLayer(&brain->CriticNet, Relu_Create(l1));
	brain->Q1 = Model_AddLayer(&brain->CriticNet, Dense_Create(1, R_XAVIER, l1));

	Layer* l2 = Model_AddLayer(&brain->CriticNet, Dense_Create(64, R_XAVIER, conc1));
	l2 = Model_AddLayer(&brain->CriticNet, Relu_Create(l2));
	l2 = Model_AddLayer(&brain->CriticNet, Dense_Create(64, R_XAVIER, l2));
	l2 = Model_AddLayer(&brain->CriticNet, Relu_Create(l2));
	brain->Q2 = Model_AddLayer(&brain->CriticNet, Dense_Create(1, R_XAVIER, l2));
	//================== target networks ====================
	brain->ActorTargetNet = Model_Create();
	brain->inpAT = Model_AddLayer(&brain->ActorTargetNet, Input_Create(brain->input_shape));
	Layer* lt = Model_AddLayer(&brain->ActorTargetNet, Dense_Create(64, R_XAVIER, brain->inpAT));
	lt = Model_AddLayer(&brain->ActorTargetNet, Relu_Create(lt));
	lt = Model_AddLayer(&brain->ActorTargetNet, Dense_Create(64, R_XAVIER, lt));
	lt = Model_AddLayer(&brain->ActorTargetNet, Relu_Create(lt));
	lt = Model_AddLayer(&brain->ActorTargetNet, Dense_Create(n_acts, R_XAVIER, lt));
	brain->actor_target = Model_AddLayer(&brain->ActorTargetNet, TanhA_Create(lt));

	brain->CriticTargetNet = Model_Create();
	brain->inpCT = Model_AddLayer(&brain->CriticTargetNet, Input_Create(brain->input_shape));//observations
	brain->inpCTA = Model_AddLayer(&brain->CriticTargetNet, Input_Create((shape) { 1, 1, n_acts}));//actions
	Layer* conc2 = Model_AddLayer(&brain->CriticTargetNet, Conc_Create(brain->inpCT, brain->inpCTA));//obs+acts

	Layer* l1t = Model_AddLayer(&brain->CriticTargetNet, Dense_Create(64, R_XAVIER, conc2));
	l1t = Model_AddLayer(&brain->CriticTargetNet, Relu_Create(l1t));
	l1t = Model_AddLayer(&brain->CriticTargetNet, Dense_Create(64, R_XAVIER, l1t));
	l1t = Model_AddLayer(&brain->CriticTargetNet, Relu_Create(l1t));
	brain->QT1 = Model_AddLayer(&brain->CriticTargetNet, Dense_Create(1, R_XAVIER, l1t));

	Layer* l2t = Model_AddLayer(&brain->CriticTargetNet, Dense_Create(64, R_XAVIER, conc2));
	l2t = Model_AddLayer(&brain->CriticTargetNet, Relu_Create(l2t));
	l2t = Model_AddLayer(&brain->CriticTargetNet, Dense_Create(64, R_XAVIER, l2t));
	l2t = Model_AddLayer(&brain->CriticTargetNet, Relu_Create(l2t));
	brain->QT2 = Model_AddLayer(&brain->CriticTargetNet, Dense_Create(1, R_XAVIER, l2t));

	brain->par = OptParams_Create();
	brain->par.method = NRMSPROP;
	brain->par.learning_rate = 0.0001f;

	brain->tau = 0.01f;
	brain->noise = initNoise(0, 0.05f, 0);
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

Tensor TD3_Forward(TD3* brain, Tensor* state)
{
	brain->inpA->input = state;
	Model_Forward(&brain->ActorNet);
	Tensor prob = Tensor_CreateCopy(&brain->actor->output);
	return prob;
}

Tensor TD3_SelectAction(TD3* brain, Tensor* state, float eps)
{
	brain->inpA->input = state;
	Model_Forward(&brain->ActorNet);
	Tensor prob = Tensor_CreateCopy(&brain->actor->output);
	
	for (size_t i = 0; i < prob.n; i++)
	{
		float noise = Clamp(getNoiseVal(&brain->noise), -brain->noise_clip, brain->noise_clip);
		float res = prob.w[i] + noise;
		res = Clamp(res, -1.f, 1.f);
		prob.w[i] = res;
	}
	
	return prob;
}

float TD3_TrainTrace(TD3* brain, Tensor* states, Tensor* last_state, Tensor* probs, float* rewards, int n, int iter)
{
	float q_total_loss = 0;
	for (int i = 0; i < n; i++)
	{
		//get actor target actions
		float done = 1.f;
		Tensor* next_state = last_state;
		if (i < n - 1) { next_state = &states[i + 1]; done = 0; }
		brain->inpAT->input = next_state;
		Model_Forward(&brain->ActorTargetNet);
		Tensor next_action = Tensor_CreateCopy(&brain->actor_target->output);
		for (size_t i = 0; i < next_action.n; i++)
		{
			float noise = Clamp(getNoiseVal(&brain->noise), -brain->noise_clip, brain->noise_clip);
			float res = next_action.w[i] + noise;
			res = Clamp(res, -1.f, 1.f);
			next_action.w[i] = res;
		}

		brain->inpCT->input = next_state;
		brain->inpCTA->input = &next_action;
		Model_Forward(&brain->CriticTargetNet);
		float qt_min = min(brain->QT1->output.w[0], brain->QT2->output.w[0]);
		Tensor target_Q = Tensor_Create(brain->QT1->output.s, qt_min);
		target_Q.w[0] = target_Q.w[0] * (1.f - done) * brain->gamma + rewards[i];

		brain->inpC->input = &states[i];
		brain->inpCA->input = &probs[i];
		Model_Forward(&brain->CriticNet);
		Tensor* current_Q1 = &brain->Q1->output;
		Tensor* current_Q2 = &brain->Q2->output;
		
		float q_loss = MSE_Loss(current_Q1, &target_Q) + MSE_Loss(current_Q2, &target_Q);
		q_total_loss += q_loss;
		Model_Backward(&brain->CriticNet);
		OptimizeModel(&brain->CriticNet, &brain->par);
		Tensor_Free(&next_action);
		Tensor_Free(&target_Q);

		//delayed update
		if (iter % brain->update_frq == 0) {
			//setup current actor
			brain->inpA->input = &states[i];
			Model_Forward(&brain->ActorNet);
			Tensor *new_actions = &brain->actor->output;

			brain->inpC->input = &states[i];
			brain->inpCA->input = new_actions;
			Model_Forward(&brain->CriticNet);
			Tensor* new_Q = &brain->Q1->output;
			Tensor* new_Q2 = &brain->Q2->output;
			//maximize main q, gradient ascent
			FillArray(new_Q->dw, new_Q->n, -1.f);
			FillArray(new_Q2->dw, new_Q2->n, 0.f);
			Model_Backward(&brain->CriticNet);
			Model_Backward(&brain->ActorNet);
			OptimizeModel(&brain->ActorNet, &brain->par);
			Model_CLearGrads(&brain->CriticNet);
			//update targets
			SoftUpdate_Targets(brain);
		}
	}
	//q_total_loss /= (float)(n);
	//if (iter % brain->update_frq == 0)
	//	printf("q loss: %f\n", q_total_loss);

	return -1.f;
}

void SoftUpdate_Targets(TD3* brain) 
{
	//printf("Soft update targets\n");
	//get grads
	dList ac = Model_getGradients(&brain->ActorNet);
	dList acT = Model_getGradients(&brain->ActorTargetNet);

	dList cr = Model_getGradients(&brain->CriticNet);
	dList crT = Model_getGradients(&brain->CriticTargetNet);
	//update
	soft_update(&acT, &ac, brain->tau);
	soft_update(&crT, &cr, brain->tau);

	dList_free(&ac);
	dList_free(&acT);
	dList_free(&cr);
	dList_free(&crT);
}
