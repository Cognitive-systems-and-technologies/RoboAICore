#pragma once

#include "geometry/TVec3.h"
#include "TCommon.h"

#include "Model.h"
#include "tList.h"
#include "fList.h"
#include "Utils.h"

#include "RL/ACBrain.h"
#include "RL/RLBrain.h"

#include "cart.h"

struct Samples 
{
    tList states;
    fList rewards;
    fList actions;
};

class Agent {
public:
    enum Phase { TRAIN, TEST };
    enum MoveDirection
    {
       Left = 0, Right = 1, Idle = 2
    };

    Cart* cart;
    float epsilon = 0.8f;
    int n_actions = 3;
    Tensor state;
    Phase phase = Phase::TEST;
    ACBrain *brain;
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
        sa.actions = fList_create();
        sa.rewards = fList_create();

        brain = ACBrain_Create(input_shape, n_actions);
    }

    ~Agent() 
    {
    }

    int Policy(Tensor* s)
    {
        Tensor t = ACBrain_Forward(brain, s);
        shape max = T_Argmax(&t);
        int act = max.d;
        Tensor_Free(&t);
        return act;
    }

    int Act(Tensor* s)
    {
        epsilon *= 0.99999f;
        if (rngFloat() <= epsilon) {
            int ra = rngInt(0, n_actions - 1);
            return ra;
        }
        else {
            Tensor t = ACBrain_Forward(brain, s);
            shape max = T_Argmax(&t);
            int act = max.d;
            Tensor_Free(&t);
            return act;
        }
    }

    float GetReward()
    {
        float reward = 1.0f;
        return reward;
    }
    float trace_reward = 0;
    void Discover()
    {
        if (phase == TRAIN) {
            int a = Act(&state);
            
            Tensor next_state = Move((MoveDirection)a);
            float reward = GetReward();
            trace_reward += reward;

            tList_push(&sa.states, &state);
            fList_push(&sa.actions, (float)a);
            fList_push(&sa.rewards, (float)reward);

            Tensor_Copy(&state, &next_state);
            Tensor_Free(&next_state);
            if (cart->needToReset())
            {
                printf("\nepsilon: %f trace reward: %f\n", epsilon, trace_reward);
                if (trace_reward > 500)
                sa.rewards.data[sa.rewards.length - 1] = 10.f;
                else
                    sa.rewards.data[sa.rewards.length - 1] = -10.f;
                ACBrain_TrainTrace(brain, sa.states.data, sa.rewards.data, sa.actions.data, sa.states.length);
                                
                tList_free(&sa.states);
                fList_free(&sa.actions);
                fList_free(&sa.rewards);

                sa.states = tList_create();
                sa.actions = fList_create();
                sa.rewards = fList_create();

                if (trace_reward > 4000)
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
            int action = Policy(&state);
            Tensor next_state = Move(MoveDirection(action));
            Tensor_Free(&state);
            state = next_state;
        }
    }
    
    Tensor Move(MoveDirection d)
    {
        cart->Step((int)d);
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
