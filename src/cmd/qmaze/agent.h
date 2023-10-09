#pragma once

#include "geometry/TVec3.h"
#include "TCommon.h"
#include "quad.h"
#include "grid.h"

#include "Model.h"
#include "tList.h"
#include "fList.h"
#include "Utils.h"

#include "RL/ACBrain.h"
#include "RL/RLBrain.h"

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
        Up = 0, Down = 1, Left = 2, Right = 3
    };

    int curI = 0;
    int curJ = 0;

    Grid* grid;
   
    float epsilon = 0.9f;
    int n_actions = 4;

    std::vector<Samples> history;

    Tensor state;
    Phase phase = Phase::TEST;
    float wallReward = 0;
    float total_reward = 0;

    ACBrain *brain;

    void Reset() 
    {
        for (size_t i = 0; i < grid->Rows; i++)
        {
            for (size_t j = 0; j < grid->Cols; j++)
            {
                grid->gridOfCells[i][j].SetVisited(false);
            }
        }
        grid->gridOfCells[9][9].quad.Color = { 1.f,0,0 };
    }
    Samples sa;
    Agent(Grid* g)
    {
        grid = g;
        quad.Rescale(grid->cellSize);
        float rc = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        float gc = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        float bc = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        quad.Color = {rc, gc, bc}; //{0.8, 0, 0};
        SetRandomPos();

        shape i_s = { grid->Rows, grid->Cols, 1 };
        state = Tensor_Create(i_s, 0);
        
        sa.states = tList_create();
        sa.actions = fList_create();
        sa.rewards = fList_create();

        //history.push_back(sa);
        brain = ACBrain_Create(i_s, n_actions);
    }

    ~Agent() 
    {
        //delete brain;
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
        if (curI == 9 && curJ == 9) return 10.f;
        float reward = -0.1f;
        return reward+wallReward;
    }
    
    void Discover()
    {
        if (phase == TRAIN) {
            int a = Act(&state);
            
            Tensor next_state = Move((MoveDirection)a);
            float reward = GetReward();
            wallReward = 0;
            total_reward += reward;
            //printf("total_reward: %f\n", total_reward);
            
            tList_push(&sa.states, &state);
            fList_push(&sa.actions, (float)a);
            fList_push(&sa.rewards, (float)reward);

            Tensor_Copy(&state, &next_state);
            Tensor_Free(&next_state);
            if (reward == 10.f || total_reward < -200.f)
            {
                printf("epsilon: %f\n", epsilon);
                if (reward == 10.f) printf("GOAL REACHED!\n");
                //if (total_reward < -200.f) history[history.size() - 1].rewards.data[history[history.size() - 1].rewards.length-1] = -1.f;
                
                //for (size_t i = 0; i < history.size(); i++)
                //{
                //Samples s = history[history.size()-1];
                ACBrain_TrainTrace(brain, sa.states.data, sa.rewards.data, sa.actions.data, sa.states.length);
                //}
                
                tList_free(&sa.states);
                fList_free(&sa.actions);
                fList_free(&sa.rewards);

                sa.states = tList_create();
                sa.actions = fList_create();
                sa.rewards = fList_create();
                //history.push_back(sa);

                total_reward = 0;
                Reset();
                SetRandomPos();
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
    
    void SetRandomPos() 
    {
        Grid::Vec2 v = grid->FindFreeRandomPosition();
        SetPos(v.x, v.y);
    }

    void SetPos(int x, int y)
    {
        int i = curI, j = curJ;
        if (x >= 0 && x <= grid->Rows) i = x;
        if (y >= 0 && y <= grid->Cols) j = y;

        Cell *cur = &grid->gridOfCells[i][j];
        cur->SetVisited(true);
        if (cur->walkable)
        {
            TVec3 c = cur->quad.Pos;
            quad.Pos = c;
            curI = i; curJ = j;
            printf("RESET POSITION\n");
        }
       
    }

    Tensor Move(MoveDirection d)
    {
        int i = curI, j = curJ;
        switch (d)
        {
        case MoveDirection::Down: curJ += 1; break;
        case MoveDirection::Left: curI -= 1; break;
        case MoveDirection::Right: curI += 1; break;
        case MoveDirection::Up: curJ -= 1; break;
        }
        if (curI < 0 || curI >= grid->Rows) { curI = i; wallReward = -0.05f; }
        if (curJ < 0 || curJ >= grid->Cols) { curJ = j; wallReward = -0.05f; }

        Cell *cur = &grid->gridOfCells[curI][curJ];

        cur->SetVisited(true);

        if (cur->walkable)
        {
            TVec3 c = cur->quad.Pos;
            quad.Pos = c;
            if (curI != i || curJ != j)
                wallReward = 0;
        }
        else
        {
            curI = i;
            curJ = j;
            wallReward = -0.05f;
        }

        return grid->ToState(curI, curJ);
    }

    void Draw()
    {
        quad.Draw();
    }

private:
    glQuad quad;
};
