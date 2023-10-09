#pragma once
#ifndef GRID_H
#define GRID_H

#include "geometry/TVec3.h"
#include "TCommon.h"
#include "cell.h"
#include <vector>
#include "Tensor.h"

class Grid{
public:
	float cellSize = 0.5f;
	std::vector<std::vector<Cell>> gridOfCells;
	int Rows = 1;
	int Cols = 1;
	float epsilon = 0.8f;//threshold for random numbers generator

	Grid(int _Rows, int _Cols)
	{
		Rows = _Rows;
		Cols = _Cols;
        generateGrid(Rows, Cols);
        gridOfCells[9][9].quad.Color = { 1.f,0,0 };
        //SimpleMaze();
	}

    Tensor ToState(int curI, int curJ) 
    {
        Tensor state = Tensor_Create({Rows, Cols, 1},0);
        for (size_t i = 0; i < Rows; i++)
        {
            for (size_t j = 0; j < Cols; j++)
            {
                Cell* c = &gridOfCells[i][j];
                if (c->walkable == false)
                    Tensor_Set(&state, i,j,0, -1.f);
            }
        }
        Tensor_Set(&state, curI, curJ, 0, 10.f);
        return state;
    }

    void generateGrid(int ic, int jc)
    {
        for (int i = 0; i < ic; i++) {
            std::vector<Cell> row;
            for (int j = 0; j < jc; j++)
            {
                TVec3 pos = {i * cellSize, j * cellSize, 0,};
                Cell c(pos, cellSize, i, j, SimpleWall());
                row.push_back(c);
            }
            gridOfCells.push_back(row);
        }
    }

    bool SimpleWall() 
    {
        bool walkable = true;
        float f = rngFloat();
        if (f > epsilon)
            walkable = false;
        return walkable;
    }

    void SimpleMaze()
    {
        float placementThreshold = .6f;
        int rMax = Rows;
        int cMax = Cols;
        for (int i = 0; i < rMax; i++)
        {
            for (int j = 0; j < cMax; j++)
            {
                
                // outside wall
                /*
                if (i == 0 || j == 0 || i == rMax || j == cMax)
                {
                    gridOfCells[i][j].SetWalkable(false);
                }
                else if (i == 1 || j == 1 || i == rMax-1 || j == cMax-1)
                {
                    gridOfCells[i][j].SetWalkable(true);
                }
                // every other inside space
                else
                    */
                if (i % 2 == 0 && j % 2 == 0)
                {
                    float f = rngFloat();
                    //float f = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                    if (f > placementThreshold)
                    {
                        (&gridOfCells[i][j])->SetWalkable(false);

                        int a = f < .5f ? 0 : (f < .5f ? -1 : 1);
                        int b = a != 0 ? 0 : (f < .5f ? -1 : 1);
                        (&gridOfCells[i+a][j+b])->SetWalkable(false);
                    }
                }
            }
        }
    }

    struct  Vec2
    {
        int x, y;
    };

    Vec2 FindFreeRandomPosition()
    {
        int rMax = Rows - 1;
        int cMax = Cols - 1;
        Vec2 pos = {-1, -1};
        //==0 free
        int isWall = 1;
        while (isWall > 0)
        {
            pos.x = rand() % (rMax - 0 + 1) + 0;
            pos.y = rand() % (cMax - 0 + 1) + 0;
            Cell c = gridOfCells[pos.x][pos.y];
            isWall = c.walkable ? 0 : 1;
        }
        return pos;
    }

	void Draw()
	{
        for (size_t i = 0; i < Rows; i++)
        {
            for (size_t j = 0; j < Cols; j++)
            {
                gridOfCells[i][j].Draw();
            }
        }
	}

private:
};
#endif