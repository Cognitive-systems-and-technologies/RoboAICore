#pragma once

#include "geometry/TVec3.h"
#include "TCommon.h"

#include "Model.h"
#include "tList.h"
#include "fList.h"
#include "Utils.h"

#include "RL/TD3.h"

#include "shape.h"
#include "box2d/box2d.h"

struct Samples 
{
    tList states;
    tList probs;
    fList rewards;
    fList actions;
};

class RBuffer
{
public:
    int capacity;
    std::vector<Samples> history;
    RBuffer(int max_length) 
    {
        this->capacity = max_length;
    }

    void PushSample(Samples s)
    {
        history.push_back(s);
        int h_size = history.size();
        if (h_size > capacity) {
            FreeSample(&history[0]);
            pop_front(this->history);
        }
    }

    int length() {return history.size();}

    template<typename T>
    void pop_front(std::vector<T>& vec)
    {
        //assert(!vec.empty());
        vec.erase(vec.begin());
    }

    void FreeSample(Samples* s)
    {
        tList_free(&s->states);
        tList_free(&s->probs);
        fList_free(&s->actions);
        fList_free(&s->rewards);
    }

    void FreeBuffer() 
    {
        for (size_t i = 0; i < history.size(); i++)
        {
            FreeSample(&history[i]);
        }
    }
private:

};

class Agent {
public:
    enum Phase { TRAIN, TEST };
    enum MoveDirection
    {
       Left = 0, Right = 1, Idle = 2
    };

    float epsilon = 0.99f;
    int n_actions = 2;
    Tensor state;
    Phase phase = Phase::TRAIN;
    TD3 *brain;
    Samples sa;
    shape input_shape = { 1, 1, 8 };

    float timeStep = 1.0f / 60.0f;
    int32 velocityIterations = 6;
    int32 positionIterations = 2;

    b2World* world = NULL;
    TDCircle* body;
    TDCircle* target;
    float bordersSize = 1.f;

    RBuffer *buffer;
    int batch_size = 128;

    Agent(b2World* w, float area, TDCircle *target)
    {
        this->world = w;
        this->target = target;
        state = Tensor_Create(input_shape, 0);
        
        sa.states = tList_create();
        sa.probs = tList_create();
        sa.actions = fList_create();
        sa.rewards = fList_create();

        brain = TD3_Create(input_shape, n_actions);
        //brain->tau = 0.01f;
        //brain->gamma = 0.95f;
        //brain->update_frq = 3;
        body = new TDCircle(w, 0, 0, 0.5f, 1.f, true);
        this->bordersSize = area;
        this->buffer = new RBuffer(4096);
    }

    ~Agent() 
    {
        this->buffer->FreeBuffer();
        delete(this->buffer);
    }

    TVec2 Policy(Tensor* s)
    {
        Tensor t = TD3_Forward(brain, s);
        printf("out shape:{ %d %d %d }\n", t.s.w, t.s.h, t.s.d);
        float x = t.w[0]; 
        float y = t.w[1];
        Tensor_Free(&t);
        return {x,y};
    }

    Tensor Act(Tensor* s)
    {
        Tensor prob = TD3_SelectAction(brain, s, epsilon);
        //int act = rng_by_prob(prob.w, prob.n);
        //Tensor_Free(&prob);
        if (epsilon > 0.05f)
            epsilon *= 0.999999f;
        return prob;
    }

    float euclidean_distance(b2Vec2 a, b2Vec2 b) 
    {
        float ab1 = (a.x - b.x);
        float ab2 = (a.y - b.y);
        float res = sqrtf(ab1*ab1+ab2*ab2);
        return res;
    }

    float GetReward()
    {
        b2Vec2 p = body->Pos();
        b2Vec2 t = target->Pos();
        float dist = (p - t).Length();
        float r = target->R;
        float reward =  (dist / (bordersSize * 2.f));
        return reward; //(dist<r)?reward+0.5f:reward/2.f;
    }

    float trace_reward = 0;
    int counter = 0;
    int step = 0;
    void Discover()
    {
        if (phase == TRAIN) {
            Tensor a = Act(&state);
            //reduce lr to save accuracy for adan opt
            float x = a.w[0];
            float y = a.w[1];
            //printf("x: %f, y: %f\n", x, y);
            Tensor next_state = MoveContinuous(x, y);
            float reward = GetReward();
            trace_reward += reward;

            tList_push(&sa.states, &state);
            tList_push(&sa.probs, &a);
            fList_push(&sa.actions, (float)0);
            fList_push(&sa.rewards, (float)reward);

            Tensor_Copy(&state, &next_state);
            Tensor_Free(&next_state);
            Tensor_Free(&a);
            if (body->needToReset(bordersSize) || sa.states.length > 500)
            {
                if (epsilon < 0.8f) { brain->par.learning_rate = 0.0000001f; printf("lr: %f\n", brain->par.learning_rate); }
                else if (epsilon < 0.9f) { brain->par.learning_rate = 0.000001f; printf("lr: %f\n", brain->par.learning_rate); }
                
                printf("\ntrace reward: %f eps: %f length: %d\n", trace_reward, epsilon, sa.states.length);
                if (body->needToReset(bordersSize))
                    sa.rewards.data[sa.rewards.length - 1] = -1.f;
                //if(trace_reward >10.f)
                //    sa.rewards.data[sa.rewards.length - 1] = 1.f;
                //else
                //sa.rewards.data[sa.rewards.length - 1] = -1.f;

                //Tensor dummy = Tensor_Create({ 1,1,n_actions }, 0);
                //tList_push(&sa.states, &state);
                //tList_push(&sa.probs, &dummy);
                //fList_push(&sa.actions, (float)0);
                //fList_push(&sa.rewards, (float)0);
                //Tensor_Free(&dummy);

                //ACBrain_TrainTrace(brain, sa.states.data, sa.rewards.data, sa.actions.data, sa.states.length);
                //for (size_t i = 0; i < history.size(); i++)
                //{
                    //Samples cs = history[i];
                if (step % 10 == 0) {
                    int len = buffer->length();
                    if (len > batch_size)
                        for (size_t i = 0; i < batch_size; i++)
                        {
                            int idx = rngInt(0, len - 1);
                            Samples s = buffer->history[idx];
                            TD3_TrainTrace(brain, s.states.data, s.probs.data, s.rewards.data, s.states.length, counter);
                            counter++;
                        }
                }
                step++;
                    //}

                
                //tList_free(&sa.states);
                //tList_free(&sa.probs);
                //fList_free(&sa.actions);
                //fList_free(&sa.rewards);
                //if(trace_reward>0)
                buffer->PushSample(sa);
                //else 
                //{
                //    tList_free(&sa.states);
                //    tList_free(&sa.probs);
                //    fList_free(&sa.actions);
                //    fList_free(&sa.rewards);
                //}
                //history.push_back(sa);
                sa = Samples();
                sa.states = tList_create();
                sa.probs = tList_create();
                sa.actions = fList_create();
                sa.rewards = fList_create();
                                
                trace_reward = 0;
                target->SetRandomPos(bordersSize);
                ResetPosition();
            }
        }
        else 
        {
            TVec2 force = Policy(&state);
            Tensor next_state = MoveContinuous(force.x, force.y);
            float rew = GetReward();
            if (rew <= 0.3f) {
                printf("reward: %f\n", rew);
            }
            Tensor_Free(&state);
            state = next_state;
        }
    }

    void ResetPosition() 
    {
        body->SetRandomPos(bordersSize);
        b2Vec2 p = body->Pos();
        b2Vec2 v = body->Vel();
        float ang = body->Angle();
        b2Vec2 t = target->Pos();
        float len = (p - t).Length();

        state.w[0] = p.x;
        state.w[1] = p.y;
        state.w[2] = v.x;
        state.w[3] = v.y;
        state.w[4] = len;
        state.w[5] = ang;

        state.w[6] = t.x;
        state.w[7] = t.y;
    }
    
    Tensor MoveContinuous(float x, float y)
    {
        //body->ApplyForce(x, y);
        //world->Step(timeStep, velocityIterations, positionIterations);
        b2Vec2 curPos = body->Pos();
        b2Vec2 newPos(curPos.x+x/2.f, curPos.y+y/2.f);
        body->m_body->SetTransform(newPos, 0);
        //body->m_body->SetAngularVelocity(0);
        //cart->ApplyForceValue(force*0.1f);
        //cart->world->Step(timeStep, velocityIterations, positionIterations);
        Tensor s = Tensor_Create(input_shape, 0);
        b2Vec2 p = body->Pos();
        b2Vec2 v = body->Vel();
        float ang = body->Angle();

        b2Vec2 t = target->Pos();
        float len = (p - t).Length();

        s.w[0] = p.x;
        s.w[1] = p.y;
        s.w[2] = v.x;
        s.w[3] = v.y;
        s.w[4] = len;
        s.w[5] = ang;

        s.w[6] = t.x;
        s.w[7] = t.y;
        return s;
    }

    void Draw() 
    {
        body->Draw();
    }

private:
};
