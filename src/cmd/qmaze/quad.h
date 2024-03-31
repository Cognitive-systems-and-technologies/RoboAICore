#pragma once
#ifndef GLQUAD_H
#define GLQUAD_H

#include "geometry/TVec3.h"
#include "TCommon.h"

#include <glad/gl.h>

class glQuad {
public:
	float width, height;

	TVec3 Color = {1.f, 0.f, 0.f};
	TVec3 Pos = {0, 0, 0};

	glQuad(float w, float h) {
		width = w;
		height = h;
		CalcList();
	}

	glQuad(TVec3 p) {
		Pos = p;
		width = 1.f;
		height = 1.f;
		CalcList();
	}

	glQuad() {
		width = 1.f;
		height = 1.f;
		CalcList();
	}

	TVec3 center() 
	{
		return { Pos.x + width * 0.5f, Pos.y + height * 0.5f, Pos.z };
	}

	void Draw()
	{
		glColor3f(Color.x, Color.y, Color.z);
		glPushMatrix();
		glTranslated(Pos.x, Pos.y, Pos.z);
		glCallList(MAP_LIST);
		//DrawQuad();
		glPopMatrix();
	}

	void Rescale(float w, float h) 
	{
		width = w;
		height = h;
		CalcList();
	}

	void Rescale(float size)
	{
		Rescale(size, size);
	}

private:
	int MAP_LIST;
	void CalcList()
	{
		MAP_LIST = glGenLists(1);
		glNewList(MAP_LIST, GL_COMPILE);
		DrawQuad();
		glEndList();
	}

	void DrawQuad()
	{
		/*
		TVec3 v0 = TVec3_Create(0, 0, -height);
		TVec3 v1 = TVec3_Create(width, 0, -height);
		TVec3 v2 = TVec3_Create(width, 0, 0);
		TVec3 v3 = TVec3_Create(0, 0, 0);
		*/
		TVec3 v0 = TVec3_Create(width, 0, 0);
		TVec3 v1 = TVec3_Create(width, height, 0);
		TVec3 v2 = TVec3_Create(0, height, 0);
		TVec3 v3 = TVec3_Create(0, 0, 0);

		//glBindTexture(GL_TEXTURE_2D, imageTexture);

		glBegin(GL_QUADS);
		glTexCoord2f(0, 0);
		glVertex3f(v0.x, v0.y, v0.z);
		glTexCoord2f(1, 0);
		glVertex3f(v1.x, v1.y, v1.z);
		glTexCoord2f(1, 1);
		glVertex3f(v2.x, v2.y, v2.z);
		glTexCoord2f(0, 1);
		glVertex3f(v3.x, v3.y, v3.z);
		glEnd();

		glColor3f(0.8f, 0.8f, 0.8f);
		glBegin(GL_LINE_LOOP);
		glVertex3f(v0.x, v0.y, v0.z);
		glVertex3f(v1.x, v1.y, v1.z);
		glVertex3f(v2.x, v2.y, v2.z);
		glVertex3f(v3.x, v3.y, v3.z);
		glEnd();

		//glBindTexture(GL_TEXTURE_2D, 0);
	}
};
#endif