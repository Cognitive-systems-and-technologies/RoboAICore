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
	brain->net = Model_Create();
	brain->inp = Model_AddLayer(&brain->net, Input_Create(brain->input_shape));
	Layer* l = Model_AddLayer(&brain->net, Dense_Create(16, R_XAVIER, brain->inp));
	l = Model_AddLayer(&brain->net, Dense_Create(16, R_XAVIER, l));
	brain->out = Model_AddLayer(&brain->net, Dense_Create(n_outputs, R_XAVIER, l));
	//l = Model_AddLayer(&brain->net, Regression_Create(l));


	//printf("Brain created");
	brain->buffer = ReplayBuffer_Create(64, 64);
	brain ->par = OptParams_Create();
	brain->par.method = ADAN;
	brain->par.learning_rate = 0.0001f;
	brain->discount = 0.95f;
	return brain;
}

void RLBrain_Record(RLBrain *brain, Tensor* state, Tensor* next_state, int action, float reward, int done)
{
	ReplayBuffer_Record(brain->buffer, state, next_state, action, reward, done);
}

Tensor RLBrain_Forward(RLBrain *brain, Tensor *state)
{
	//Tensor *y = Seq_Forward(&brain->net, state, 0);
	brain->inp->input = state;
	Model_Forward(&brain->net);
	return brain->out->output;
}

float RLBrain_Train(RLBrain *brain)
{
	/*
	if (brain->buffer->buffer->length >= brain->buffer->batch_size) {
		float cur_loss = 0.0f;
		for (int i = brain->buffer->buffer->length-(int)1; i > 0; i--)
		{
			Sample *s = (Sample*)brain->buffer->buffer->data[i].elem;
			Tensor y = Tensor_Create((shape){1,1,2}, 0);
			y.w[0] = (float)s->action;
			if(s->done)
				y.w[1] = s->reward;
			else 
			{
				Tensor* next = brain->net.NetForward(&brain->net, s->next_state);
				float Q_sa = T_MaxValue(next);
				y.w[1] = s->reward + brain->discount * Q_sa;
			}
			Optimize(&brain->net, &brain->par, s->state, &y);
			LData* d = (LData*)brain->net.Layers[brain->net.n_layers - 1]->aData;
			cur_loss += d->loss;
			Tensor_Free(&y);
		}
		float loss = cur_loss / brain->buffer->batch_size;
		return loss;
	}
	*/
	return -1.f;
}

float RLBrain_TrainTrace(RLBrain* brain, Tensor* states, float* rewards, float* actions, int n)
{
	float total_loss = 0;
	for (int i = 0; i < n-1; i++)
	{
		brain->inp->input = &states[i+1];
		Model_Forward(&brain->net);
		float Q_sa = T_MaxValue(&brain->out->output);
		float target = rewards[i] + brain->discount * Q_sa;
		target = (i == n - 2) ? rewards[i] : target;

		brain->inp->input = &states[i];
		Model_Forward(&brain->net);
		float loss = Regression_Loss(&brain->out->output, (int)actions[i], target);
		total_loss += loss;
		Model_Backward(&brain->net);
		OptimizeModel(&brain->net, &brain->par);
	}
	total_loss /= (float)n;
	printf("trace_loss: %f\n", total_loss);
	return total_loss;
}
