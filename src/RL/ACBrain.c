#include "ACBrain.h"

ACBrain* ACBrain_Create(shape state_shape, int n_outputs)
{
	ACBrain* brain = malloc(sizeof(ACBrain));
	if (!brain)
	{
		return NULL;
	}
	brain->input_shape = (shape){ state_shape.w, state_shape.h, state_shape.d };
	brain->num_outputs = n_outputs;
	//brain->net = ACBrain_CreateNet(brain->input_shape, brain->num_outputs);
	brain->gamma = 0.99f;
	brain->discount = 0.01f;
	//create net
	brain->ActorNet = Model_Create();
	brain->inpA = Model_AddLayer(&brain->ActorNet, Input_Create(brain->input_shape));
	Layer* l = Model_AddLayer(&brain->ActorNet, Dense_Create(64, R_XAVIER, brain->inpA));
	l = Model_AddLayer(&brain->ActorNet, Dense_Create(64, R_XAVIER, l));
	brain->actor = Model_AddLayer(&brain->ActorNet, Dense_Create(n_outputs, R_XAVIER, l));

	brain->CriticNet = Model_Create();
	brain->inpC = Model_AddLayer(&brain->CriticNet, Input_Create(brain->input_shape));
	Layer* l2 = Model_AddLayer(&brain->CriticNet, Dense_Create(64, R_XAVIER, brain->inpC));
	l2 = Model_AddLayer(&brain->CriticNet, Dense_Create(64, R_XAVIER, l2));
	brain->critic = Model_AddLayer(&brain->CriticNet, Dense_Create(1, R_XAVIER, l2));

	brain->par = OptParams_Create();
	brain->par.method = ADAN;
	brain->par.learning_rate = 0.00001f;
	brain->I = 1.f;
	return brain;
}

Tensor ACBrain_Forward(ACBrain* brain, Tensor* state)
{
	brain->inpA->input = state;
	Model_Forward(&brain->ActorNet);
	Tensor prop = SoftmaxProb(&brain->actor->output);//need to be free() after use
	return prop;
}

float ACBrain_TrainTrace(ACBrain* brain, Tensor* states, float* rewards, float* actions, int n)
{
	float* adv_rewards = createFloatArray(n);
	float discounted_sum = 0.f;
	for (int i = n - 1; i >= 0; i--)
	{
		discounted_sum = rewards[i] + brain->gamma * discounted_sum;
		adv_rewards[i] = discounted_sum;
	}
	//NormalizeArray(adv_rewards, n);

	float total_actor_loss = 0;
	float total_critic_loss = 0;
	for (int i = 0; i < n; i++)
	{
		//setup critic
		brain->inpC->input = &states[i];
		Model_Forward(&brain->CriticNet);
		float critic_value = brain->critic->output.w[0];
		float advantage = adv_rewards[i] - critic_value;
		Tensor critic_true = Tensor_Create((shape) { 1, 1, 1 }, adv_rewards[i]);
		float critic_loss = MSE_Loss(&brain->critic->output, &critic_true);
		total_critic_loss += critic_loss;
		Tensor_Free(&critic_true);
		Model_Backward(&brain->CriticNet);
		OptimizeModel(&brain->CriticNet, &brain->par);

		//setup actor
		brain->inpA->input = &states[i];
		Model_Forward(&brain->ActorNet);
		float actor_loss = Cross_entropy_Loss(&brain->actor->output, actions[i]);
		total_actor_loss += actor_loss;

		Tensor prop = SoftmaxProb(&brain->actor->output);
		for (size_t j = 0; j < brain->actor->output.n; j++)
		{
			//float y_true = (j == (int)actions[i]) ? 1.f : 0.f;
			//float der = -(y_true - prop.w[j]);
			//brain->actor->output.dw[j] = brain->discount * advantage * der;
			brain->actor->output.dw[j] *= advantage*brain->I;
		}
		Tensor_Free(&prop);
		Model_Backward(&brain->ActorNet);
		//==================
		OptimizeModel(&brain->ActorNet, &brain->par);
		//brain->I *= 0.9999f;
	}
	total_actor_loss /= n;
	total_critic_loss /= n;

	printf("actor_loss: %f, critic_loss: %f\n", total_actor_loss, total_critic_loss);
	free(adv_rewards);
	return -1.f;
}
