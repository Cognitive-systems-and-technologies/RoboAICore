#pragma once
#include "box2d/box2d.h"
#include "GLFW/glfw3.h"
#include <set>
#include <vector>
#include <string>
#include <sstream>
#include "geometry/TVec2.h"
#include "geometry/TVec3.h"

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

        glColor3f(0.8f, 0.5f, 0.8f);
        glBegin(GL_QUADS);
        for (size_t i = 0; i < polygonShape.m_count; i++)
        {
            glVertex2f(polygonShape.m_vertices[i].x, polygonShape.m_vertices[i].y);
        }
        glEnd();
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

class TDCircle 
{
public:
    b2Body* m_body;
    b2CircleShape circleShape;
    int vertexCount = 32;
    float R = 1.f;
    TVec3 color;
    float color_alpha = 0.8f;

    TDCircle(b2World* world, float x, float y, float r, float den, bool isSensor) 
    {
        b2BodyDef bodyDef;
        bodyDef.type = b2_dynamicBody;
        bodyDef.linearDamping = 3;
        m_body = world->CreateBody(&bodyDef);

        circleShape.m_p.Set(x, y);
        circleShape.m_radius = r;
        this->R = r;

        b2FixtureDef fixtureDef;
        fixtureDef.shape = &circleShape;
        fixtureDef.density = den;
        fixtureDef.friction = 0.3f;
        fixtureDef.isSensor = isSensor;
        b2Fixture* fixture = m_body->CreateFixture(&fixtureDef);

        SetPos({x,y});
        SetRandomColor();
    }

    void SetPos(b2Vec2 pos)
    {
        m_body->SetTransform(pos, 0);
        m_body->SetAngularVelocity(0);
        m_body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
    }

    void SetRandomColor() 
    {
        float rc = rngFloat();
        float gc = rngFloat();
        float bc = rngFloat();
        this->color = { rc, gc, bc }; //{0.8, 0, 0};
    }

    void ApplyForce(float x, float y) 
    {
        m_body->ApplyLinearImpulseToCenter(b2Vec2(x, y), true);
        m_body->SetAngularVelocity(0);
    }

    bool needToReset(float rectSize)
    {
        b2Vec2 pos = Pos();
        if (pos.x<-rectSize||pos.x>rectSize)
            return true;
        if (pos.y<-rectSize || pos.y>rectSize)
            return true;
        return false;
    }

    void SetRandomPos(float rectSize) 
    {
        int x = rngInt(-rectSize+1, rectSize-1);
        int y = rngInt(-rectSize+1, rectSize-1);

        m_body->SetTransform(b2Vec2(x, y), 0);
        m_body->SetAngularVelocity(0);
        m_body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
    }

    b2Vec2 Pos()
    {
        return m_body->GetPosition();
    }
    
    b2Vec2 Vel()
    {
        return m_body->GetLinearVelocity();
    }
    float AVel()
    {
        return m_body->GetAngularVelocity();
    }
    float Angle() 
    {
        return m_body->GetAngle();
    }

    TVec2 PointOnCircle(float radius, float angleInDegrees, TVec2 origin)
    {
        float x = (radius * cosf(angleInDegrees * PI / 180.f)) + origin.x;
        float y = (radius * sinf(angleInDegrees * PI / 180.f)) + origin.y;
        return { x, y };
    }

    void DrawCircle() 
    {
        glLineWidth(2.f);
        float step = 360.f / (float)vertexCount;
        //glColor3f(color.x, color.y, color.z);
        glColor4f(color.x, color.y, color.z, this->color_alpha);
        glBegin(GL_POLYGON);
        for (float i = 0; i < 360.f; i += step)
        {
            TVec2 v = PointOnCircle(this->R, i, { 0,0 });
            glVertex2f(v.x, v.y);
        }
        glEnd();
        glColor4f(0, 0, 0, this->color_alpha*0.5f);
        glBegin(GL_LINE_LOOP);
        for (float i = 0; i < 360.f; i += step)
        {
            TVec2 v = PointOnCircle(this->R, i, {0,0});
            glVertex2f(v.x, v.y);
        }
        glEnd();
        glLineWidth(1);
    }

    void Draw() 
    {
        glPushMatrix();
        glTranslatef(m_body->GetPosition().x, m_body->GetPosition().y, 0);
        glRotatef(m_body->GetAngle() * (180.f / PI), 0, 0, 1);
        
        DrawCircle();
        glPopMatrix();
    }
};