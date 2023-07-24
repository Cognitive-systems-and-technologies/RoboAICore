#include "ACBrain.h"

ACBrain *ACBrain_Create(shape state_shape, int n_outputs)
{
	ACBrain *brain = malloc(sizeof(ACBrain));
	if(!brain)
	{
		return NULL;
	}
	brain->input_shape = (shape){state_shape.w, state_shape.h, state_shape.d};
	brain->num_outputs = n_outputs;
	brain->net = ACBrain_CreateNet(brain->input_shape, brain->num_outputs);
	//printf("Brain created");
	brain->buffer = ReplayBuffer_Create(64, 64);
	brain ->par = (OptParams){ 0.1f, 1.0f, 0.001f, 32, ADAGRAD, 1.0f };
	brain->discount = 0.95f;
	brain->critic_loss_weight = 0.5f;
	brain->actor_loss_weight = 1.0f;
	brain->entropy_loss_weight = 0.05f;
	return brain;
}

void ACBrain_Record(ACBrain *brain, Tensor* state, Tensor* next_state, int action, float reward, int done)
{
	ReplayBuffer_Record(brain->buffer, state, next_state, action, reward, done);
}

Tensor* ACBrain_Forward(ACBrain *brain, Tensor *state)
{
	Tensor *y = Seq_Forward(&brain->net, state, 0);
	return y;
}

Model ACBrain_CreateNet(shape input_sh, int n_outputs)
{
	Model n = Model_Create();
	Layer* l = Model_AddLayer(&n, Input_Create(input_sh));
	l = Model_AddLayer(&n, Dense_Create(16, R_XAVIER, l));
	l = Model_AddLayer(&n, Dense_Create(16, R_XAVIER, l));

	Layer *actor = Model_AddLayer(&n, Dense_Create(n_outputs, R_XAVIER, l));
	actor = Model_AddLayer(&n, Regression_Create(actor));

	Layer *critic = Model_AddLayer(&n, Dense_Create(1, R_XAVIER, l));
	critic = Model_AddLayer(&n, Regression_Create(critic));
	return n;
}
float ACBrain_Train(ACBrain *brain)
{
	//todo fix code
	return -1.f;
}
