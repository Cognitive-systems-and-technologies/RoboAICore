#pragma once
#include "box2d/box2d.h"
#include "GLFW/glfw3.h"
#include <set>
#include <vector>
#include <string>
#include <sstream>

#define PI 3.141592653589793f
#define DEGTORAD 0.0174532925199432957f

class TDBox {
public:
    b2Body* m_body;
    b2PolygonShape polygonShape;
        
    TDBox(b2World* world, float w, float h) {
        b2BodyDef bodyDef;
        bodyDef.type = b2_dynamicBody;
        bodyDef.linearDamping = 3;
        m_body = world->CreateBody(&bodyDef);
        
        polygonShape.SetAsBox(w, h);

        b2FixtureDef fixtureDef;
        fixtureDef.shape = &polygonShape;
        fixtureDef.density = 1.0f;
        fixtureDef.friction = 0.3f;

        b2Fixture* fixture = m_body->CreateFixture(&fixtureDef);
    }

    TDBox(b2World* world, float w, float h, b2Vec2 pos, b2BodyType btype, bool isSensor, float den)
    {
        b2BodyDef bodyDef;
        bodyDef.type = btype;
        bodyDef.linearDamping = 3;
        bodyDef.angularDamping = 3;
        
        m_body = world->CreateBody(&bodyDef);

        polygonShape.SetAsBox(w, h);

        b2FixtureDef fixtureDef;
        fixtureDef.shape = &polygonShape;
        fixtureDef.density = den;
        fixtureDef.friction = 0.3f;
        fixtureDef.isSensor = isSensor;
        
        b2Fixture* fixture = m_body->CreateFixture(&fixtureDef);
        SetPos(pos);
    }

    void SetPos(b2Vec2 pos) 
    {
        m_body->SetTransform(pos, 0);
    }

    ~TDBox() {
        m_body->GetWorld()->DestroyBody(m_body);
    }

    void Draw()
    {
        glPushMatrix();
        glTranslatef(m_body->GetPosition().x, m_body->GetPosition().y, 0);
        glRotatef(m_body->GetAngle() * (180.f / PI), 0, 0, 1);
        glColor3f(0, 0, 0);
        glBegin(GL_LINE_LOOP);
        for (size_t i = 0; i < polygonShape.m_count; i++)
        {
            glVertex2f(polygonShape.m_vertices[i].x, polygonShape.m_vertices[i].y);
        }
        glEnd();
        glPopMatrix();
    }
};
