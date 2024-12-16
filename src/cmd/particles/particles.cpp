#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "linmath.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include "box2d/box2d.h"
#include "shape.h"
#include "agent.h"

#include "TCommon.h"

GLFWwindow* CreateGLFWindow(int w, int h)
{
    GLFWwindow* window = nullptr;

    if (!glfwInit())
        printf("Unable to initialize GLFW");

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    int sWidth = mode->width;
    int sHeight = mode->height;

    window = glfwCreateWindow(w, h, "Env", NULL, NULL);
    if (!window) {
        glfwTerminate();
        printf("Unable to create GLFW window");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(0.1);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);
    glEnable(GL_BLEND);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_DOUBLEBUFFER);
    glEnable(GL_DEPTH);
    return window;
}

int stepsCount = 0;
float total_reward = 0;
float alpha = 0.8f;
int trained_steps = 0;

Agent* a1, *a2;
float viewRectSize = 10.f;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        a1->ResetPosition();
        a1->target->SetRandomPos(viewRectSize);
    }
    if (key == GLFW_KEY_E && action == GLFW_PRESS) {
        printf("E pressed\n");
        a1->phase = a1->phase == Agent::Phase::TRAIN ? Agent::Phase::TEST : Agent::Phase::TRAIN;
        if (a1->phase == Agent::Phase::TEST) {
            glfwSwapInterval(1);
            printf("TEST PHASE\n");
        }
        else {
            glfwSwapInterval(0.1);
            printf("TRAIN PHASE\n");
        }
    }
}

int main() 
{
    b2Vec2 gravity(0.0f, 0.0f);
    b2World world(gravity);
    float timeStep = 1.0f / 60.0f;
    int32 velocityIterations = 6;
    int32 positionIterations = 2;

    int width = 640, height = 640;
    GLFWwindow* window = CreateGLFWindow(width, height);
    //int width, height;
    float aspect = (float)width / (float)height;
    
    float right = viewRectSize * aspect, top = viewRectSize;
    float bottom = -viewRectSize, left = -viewRectSize * aspect;
    glClearColor(1.f, 1.f, 1.f, 0.0f);

    mat4x4 m, p, mvp;
    mat4x4_ortho(p, left, right, bottom, top, 1.f, -1.f);

    glfwSetKeyCallback(window, key_callback);

    float lastFrame = 0;
    float deltaTime = 0;

    float time = 0;
    glfwSetKeyCallback(window, key_callback);
    printf("=== MA PARTICLES ENVIRONMENT ===\n");
    printf("=== Controls: R-reset, E-toogle test/train mode LMB - set agent position ===\n");
    
    //TDCircle circle(&world, 0, 0, 0.5f, 1.f, false);
    //TDCircle circle2(&world, 0, 0, 0.5f, 1.f, false);

    TDCircle Target1(&world, 5.f, -5.f, 1.5f, 1.f, true);
    TDCircle Target2(&world, -5.f, 5.f, 1.0f, 1.f, true);

    Agent agent1(&world, viewRectSize, &Target1);
    Agent agent2(&world, viewRectSize, &Target2);

    a1 = &agent1;
    a2 = &agent2;
    
    while (!glfwWindowShouldClose(window))
    {
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadMatrixf((const GLfloat*)p);
        //draw axis line
        glColor3f(0.7, 0.7, 0.7);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0.0);
        glVertex3f(right, 0, 0.0);
        glVertex3f(0, 0, 0.0);
        glVertex3f(left, 0, 0.0);
        glVertex3f(0, 0, 0.0);
        glVertex3f(0, top, 0.0);
        glVertex3f(0, 0, 0.0);
        glVertex3f(0, bottom, 0.0);
        glEnd();
        
        agent1.Discover();
        //agent2.Discover();
        //circle.ApplyForce(rngNormal(), rngNormal());
        //circle2.ApplyForce(rngNormal(), rngNormal());
        
        //circle.Draw();
        //circle2.Draw();
        agent1.Draw();
        agent2.Draw();
        //if (circle.needToReset(viewRectSize) || circle2.needToReset(viewRectSize)) { circle.SetRandomPos(viewRectSize); circle2.SetRandomPos(viewRectSize);
        //}
        //if (circle2.needToReset(viewRectSize)) circle2.SetRandomPos(viewRectSize);
        
        Target1.Draw();
        Target2.Draw();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
	return 0;
}