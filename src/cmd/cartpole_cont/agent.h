#pragma once

#include "geometry/TVec3.h"
#include "TCommon.h"

#include "Model.h"
#include "tList.h"
#include "fList.h"
#include "Utils.h"

#include "RL/DDPG.h"

#include "cart.h"

struct Samples 
{
    tList states;
    tList probs;
    fList rewards;
    fList actions;
};

class Agent {
public:
    enum Phase { TRAIN, TEST };

    Cart* cart;
    float epsilon = 0.8f;
    int n_actions = 1;
    Tensor state;
    Phase phase = Phase::TEST;
    DDPG *brain;
    Samples sa;
    shape input_shape = { 1, 1, 4 };

    float timeStep = 1.0f / 60.0f;
    int32 velocityIterations = 6;
    int32 positionIterations = 2;

    Agent(Cart* c)
    {
        cart = c;
        state = Tensor_Create(input_shape, 0);
        
        sa.states = tList_create();
        sa.probs = tList_create();
        sa.actions = fList_create();
        sa.rewards = fList_create();

        brain = DDPG_Create(input_shape, n_actions);
    }

    ~Agent() 
    {
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

            tList_push(&sa.states, &state);
            tList_push(&sa.probs, &a);
            fList_push(&sa.actions, (float)force);
            fList_push(&sa.rewards, (float)reward);

            Tensor_Copy(&state, &next_state);
            Tensor_Free(&next_state);
            Tensor_Free(&a);
            if (cart->needToReset()||sa.states.length>4000)
            {
                printf("\ntrace reward: %f eps: %f\n", trace_reward, epsilon);
                if (trace_reward > 500)
                    sa.rewards.data[sa.rewards.length - 1] = 10.f;
                else
                    sa.rewards.data[sa.rewards.length - 1] = -10.f;
                DDPG_TrainTrace(brain, sa.states.data, sa.probs.data, sa.rewards.data, sa.states.length, counter);
                counter++;

                tList_free(&sa.states);
                tList_free(&sa.probs);
                fList_free(&sa.actions);
                fList_free(&sa.rewards);

                sa.states = tList_create();
                sa.probs = tList_create();
                sa.actions = fList_create();
                sa.rewards = fList_create();
                
                if (trace_reward > 2000)
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
        cart->ApplyForceValue(force*0.1f);
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
