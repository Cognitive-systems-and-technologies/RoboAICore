#pragma once

#include "geometry/TVec3.h"
#include "TCommon.h"

#include "Model.h"
#include "tList.h"
#include "fList.h"
#include "Utils.h"

#include "RL/DDPG.h"

#include "cart.h"
#include "RL/SimpleDeque.h"

struct Samples
{
    Samples(shape s)
    {
        states = tList_create();
        probs = tList_create();
        actions = fList_create();
        rewards = fList_create();
        last_state = Tensor_Create(s, 0);
    }

    void AddSample(Tensor* state, Tensor* prob, float action, float reward)
    {
        tList_push(&states, state);
        tList_push(&probs, prob);
        fList_push(&actions, action);
        fList_push(&rewards, reward);
    }

    tList states;
    tList probs;
    fList rewards;
    fList actions;
    Tensor last_state;
};

void freeSample(void* s)
{
    Samples* sa = (Samples*)s;
    tList_free(&sa->states);
    tList_free(&sa->probs);
    fList_free(&sa->actions);
    fList_free(&sa->rewards);
    Tensor_Free(&sa->last_state);
}

class Agent {
public:
    enum Phase { TRAIN, TEST };

    Cart* cart;
    float epsilon = 0.9f;
    int n_actions = 1;
    Tensor state;
    Phase phase = Phase::TEST;
    DDPG* brain;
    Samples* sa;
    shape input_shape = { 1, 1, 4 };

    float timeStep = 1.0f / 60.0f;
    int32 velocityIterations = 6;
    int32 positionIterations = 2;

    //replay buffer:
    SimpleDeque* history = createDeque(20000);
    int batch_size = 64;

    Agent(Cart* c)
    {
        cart = c;
        state = MoveByForce(0.f); //Tensor_Create(input_shape, 0);
        sa = new Samples(input_shape);
        brain = DDPG_Create(input_shape, n_actions);
    }

    ~Agent()
    {
        printf("Clean up actor's data...\n");
        freeDeque(history, freeSample);
    }

    float Policy(Tensor* s)
    {
        Tensor t = DDPG_Forward(brain, s);
        float force = t.w[0];
        Tensor_Free(&t);
        return force;
    }

    Tensor Act(Tensor* s)
    {
        Tensor prob = DDPG_SelectAction(brain, s, epsilon);
        if (epsilon > 0.05f)
            epsilon *= 0.99999f;
        return prob;
    }

    float GetReward()
    {
        float reward = 1.0f;
        return reward;
    }

    float trace_reward = 0;
    int counter = 0;
    void Discover()
    {
        if (phase == TRAIN) {
            Tensor a = Act(&state);

            float force = a.w[0];

            Tensor next_state = MoveByForce(force);
            float reward = GetReward();
            trace_reward += reward;

            sa->AddSample(&state, &a, force, reward);

            Tensor_Copy(&state, &next_state);
            Tensor_Free(&next_state);
            Tensor_Free(&a);
            if (cart->needToReset() || sa->states.length > 5000)
            {
                if (sa->states.length < 5000)
                    sa->rewards.data[sa->rewards.length - 1] = -1;
                printf("\ntrace reward: %f eps: %f\n", trace_reward, epsilon);
                if (history->length > batch_size) {
                    for (size_t i = 0; i < batch_size; i++)
                    {
                        int k = rngInt(0, history->length - 1);
                        Samples* s = (Samples*)history->data[k].elem;
                        DDPG_TrainTrace(brain, s->states.data, &s->last_state, s->probs.data, s->rewards.data, s->states.length, counter);
                        counter++;
                    }
                }
                Tensor_CopyData(&sa->last_state, &state);
                dequeAppend(history, sa, freeSample);
                sa = new Samples(input_shape);

                if (trace_reward > 5000)
                {
                    printf("Maximum score reached, set TEST phase\n");
                    glfwSwapInterval(1.0);
                    phase = TEST;
                }

                trace_reward = 0;
                cart->Reset();
            }
        }
        else
        {
            float force = Policy(&state);
            Tensor next_state = MoveByForce(force);
            Tensor_Free(&state);
            state = next_state;
        }
    }

    Tensor MoveByForce(float force)
    {
        cart->ApplyForceValue(force * 0.1f);
        cart->world->Step(timeStep, velocityIterations, positionIterations);
        Tensor s = Tensor_Create(input_shape, 0);
        s.w[0] = cart->cartPos();
        s.w[1] = cart->cartVel();
        s.w[2] = cart->poleAngle();
        s.w[3] = cart->poleVel();
        return s;
    }

private:
};
