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
	Tensor *y = Seq_Forward(brain->net, state, 0);
	return y;
}

Net *ACBrain_CreateNet(shape input_sh, int n_outputs)
{
	Net *n = (Net*)malloc(sizeof(Net));
	if(!n)
	{
		//printf("Net allocation error");
		return NULL;
	}
	n->n_layers = 7;
	n->Layers = (Layer**)malloc(sizeof(Layer*)*n->n_layers);
	if (!n->Layers)
	{
		//printf("Layers allocation error!");
		return NULL;
	}

	n->Layers[0] = Input_Create(input_sh);
	n->Layers[1] = Dense_Create(8, n->Layers[0]->out_shape);
	n->Layers[2] = Dense_Create(8, n->Layers[1]->out_shape);

	n->Layers[3] = Dense_Create(n_outputs, n->Layers[2]->out_shape);
	n->Layers[4] = Regression_Create(n->Layers[3]->out_shape);//policy logits

	n->Layers[5] = Dense_Create(1, n->Layers[2]->out_shape);
	n->Layers[6] = Regression_Create(n->Layers[5]->out_shape);//value

	n->NetForward = Seq_Forward;
	n->NetBackward = Seq_Backward;

	return n;
}
float ACBrain_Train(ACBrain *brain)
{
	//todo test
	return -1.f;
}
