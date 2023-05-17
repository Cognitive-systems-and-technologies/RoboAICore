#include "RLBrain.h"

RLBrain *RLBrain_Create(shape state_shape, int n_outputs)
{
	RLBrain *brain = malloc(sizeof(RLBrain));
	if(!brain)
	{
		return NULL;
	}
	brain->input_shape = (shape){state_shape.w, state_shape.h, state_shape.d};
	brain->num_outputs = n_outputs;
	brain->net = RLBrain_CreateNet(brain->input_shape, brain->num_outputs);
	//printf("Brain created");
	brain->buffer = ReplayBuffer_Create(64, 64);
	brain ->par = (OptParams){ 0.1f, 1.0f, 0.001f, 32, ADAGRAD, 1.0f };
	brain->discount = 0.95f;
	return brain;
}

void RLBrain_Record(RLBrain *brain, Tensor* state, Tensor* next_state, int action, float reward, int done)
{
	ReplayBuffer_Record(brain->buffer, state, next_state, action, reward, done);
}

Tensor* RLBrain_Forward(RLBrain *brain, Tensor *state)
{
	Tensor *y = Seq_Forward(brain->net, state, 0);
	return y;
}

Net *RLBrain_CreateNet(shape input_sh, int n_outputs)
{
	Net *n = (Net*)malloc(sizeof(Net));
	if(!n)
	{
		//printf("Net allocation error");
		return NULL;
	}
	n->n_layers = 5;
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
	n->Layers[4] = Regression_Create(n->Layers[3]->out_shape);

	n->NetForward = Seq_Forward;
	n->NetBackward = Seq_Backward;

	return n;
}
float RLBrain_Train(RLBrain *brain)
{
	if (brain->buffer->buffer->length >= brain->buffer->batch_size) {
		float cur_loss = 0.0f;
		for (int i = brain->buffer->buffer->length-(int)1; i > 0; i--)
		{
			Sample *s = (Sample*)brain->buffer->buffer->data[i].elem;
			Tensor* y = Tensor_Create((shape){1,1,2}, 0, 0);
			y->w[0] = (float)s->action;
			if(s->done)
				y->w[1] = s->reward;
			else 
			{
				Tensor* next = brain->net->NetForward(brain->net, s->next_state, 0);
				float Q_sa = T_MaxValue(next);
				y->w[1] = s->reward + brain->discount * Q_sa;
			}
			Info info = Optimize(brain->net, &brain->par, s->state, y);
			cur_loss += info.cost_loss;
			Tensor_Free(y);
		}
		float loss = cur_loss / brain->buffer->batch_size;
		return loss;
	}
	return -1.f;
}
