#pragma once
#include "box2d/box2d.h"
#include "GLFW/glfw3.h"
#include <set>
#include <vector>
#include <string>
#include "shapes.h"
#include "TCommon.h"

class Cart {
public:
    TDBox *b1;
    TDBox *b2; 
    TDBox *b3;

    b2World* world;

    Cart(b2World* _world) {
        world = _world;
        b1 =new TDBox(_world, 0.04f, 0.32f, { 0.0f, 0.32f }, b2_dynamicBody, false, 0.01f);
        b2 =new TDBox(_world, 0.16f, 0.08f, { 0.0f, 0.0f }, b2_dynamicBody, false, 1.f);
        b3 =new TDBox(_world, 2.0f, 0.01f, { 0.0f, 0.f }, b2_kinematicBody, true, 1.f);

        b2RevoluteJointDef jointDef;
        jointDef.bodyA = b2->m_body;
        jointDef.bodyB = b1->m_body;
        //jointDef.collideConnected = false;
        jointDef.localAnchorB = b2Vec2(0, -0.32f);
        _world->CreateJoint(&jointDef);

        b2PrismaticJointDef prjointDef;
        prjointDef.bodyA = b3->m_body;
        prjointDef.bodyB = b2->m_body;
        prjointDef.collideConnected = false;
        _world->CreateJoint(&prjointDef);
    }

    float poleAngle() 
    {
        return b1->m_body->GetAngle();
    }

    float cartPos() 
    {
        return b2->m_body->GetPosition().x;
    }

    float cartVel() 
    {
        return b2->m_body->GetLinearVelocity().x;
    }

    float poleVel() 
    {
        return b1->m_body->GetAngularVelocity();
    }

    bool needToReset() 
    {
        float pos = cartPos();
        float angle = poleAngle();
        if (angle < -1.f || angle>1.f || pos < -1.5f || pos>1.5f)
            return true;
        return false;
    }

    void pushLeft() 
    {
        b2->m_body->ApplyLinearImpulseToCenter(b2Vec2(-0.02f, 0), true);
    }
    void pushRight()
    {
        b2->m_body->ApplyLinearImpulseToCenter(b2Vec2(0.02f, 0), true);
    }

    void Reset() 
    {
        int r = rngInt(0, 1);
        float angle = r > 0 ? 0.1f : -0.1f;
        b1->m_body->SetTransform(b2Vec2(0.0f, 0.32f), angle);
        b2->m_body->SetTransform(b2Vec2(0.0f, 0.0f), 0);
        
        b1->m_body->SetAngularVelocity(0);
        b1->m_body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));

        b2->m_body->SetAngularVelocity(0);
        b2->m_body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
    }

    void Step(int action) 
    {
        switch (action)
        {
        case 0: pushLeft(); break;
        case 1: pushRight(); break;
        default:
            break;
        }
    }

    void ApplyForceValue(float force)
    {
        b2->m_body->ApplyLinearImpulseToCenter(b2Vec2(force, 0), true);
    }

    ~Cart() {
        delete(b1);
        delete(b2);
        delete(b3);
    }

    void Draw()
    {
        b1->Draw();
        b2->Draw();
        b3->Draw();
    }
};
