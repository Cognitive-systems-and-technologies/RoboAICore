#pragma once
#ifndef CELL_H
#define CELL_H

#include "geometry/TVec3.h"
#include "TCommon.h"
#include "quad.h"

class Cell {
public:
	int i = 0;
	int j = 0;

	bool walkable = true;
	bool visited = false;
	glQuad quad;

	Cell(TVec3 pos, float cellSize, int _i, int _j, bool _walkable)
	{
		quad.Pos = pos;
		quad.width = cellSize;
		quad.height = cellSize;
		i = _i;
		j = _j;
		walkable = _walkable;
		if (walkable) quad.Color = {0.9f, 0.9f, 0.9f}; else quad.Color = { 0.3f, 0.3f, 0.3f};
	}

	void SetWalkable(bool w) 
	{
		walkable = w;
		if (walkable) quad.Color = { 0.9f, 0.9f, 0.9f }; else quad.Color = { 0.1f, 0.1f, 0.1f };
	}

	void SetVisited(bool w)
	{
		if (walkable) {
			visited = w;
			if (visited) quad.Color = { 0.7f, 0.7f, 0.7f }; else quad.Color = { 0.9f, 0.9f, 0.9f };
		}
	}

	TVec3 center()
	{
		return quad.center();
	}

	void Draw() 
	{
		quad.Draw();
	}

//private:
};
#endif