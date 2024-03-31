#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "linmath.h"

#include <stdio.h>
#include <iostream>
#include <vector>

#include "grid.h"
#include "agent.h"

Agent* link;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        printf("RESET pressed\n");
        link->epsilon = 0.5f;
        link->Reset();
        link->SetRandomPos();
    }
    if (key == GLFW_KEY_E && action == GLFW_PRESS) {
        printf("E pressed\n");
        link->phase = link->phase == Agent::Phase::TRAIN ? Agent::Phase::TEST : Agent::Phase::TRAIN;
        if (link->phase == Agent::Phase::TEST) {
            glfwSwapInterval(1);
            printf("TEST PHASE\n");
        }
        else {
            glfwSwapInterval(0.1);
            printf("TRAIN PHASE\n");
        }
    }
}

void mouse_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        double x;
        double y;
        int stepX = 64;
        glfwGetCursorPos(window, &x, &y);
        //printf("MX: %f, MY: %f\n", x, y);
        int wX = (int)(x / stepX);
        int wY = (int)(y / stepX);
        link->Reset();
        link->SetPos(wX, wY);

        printf("MX: %d, MY: %d\n", wX, wY);
    }
}

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

    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);  // Enabled Multisample 
    //glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);
    return window;
}

int main() 
{
    GLFWwindow* window = CreateGLFWindow(640, 640);
    //int width, height;
    float loss = 0;
    float right = 5.f, top = 5.f;
    float bottom = 0.f, left = 0.f;
    glClearColor(1.f, 1.f, 1.f, 0.0f);

    int numAgents = 1;
    int numItems = 0;
    int gridSize = 10;

    Grid grid(gridSize, gridSize);

    Agent agent(&grid);
    link = &agent;

    printf("Agents count: %d\n", numAgents);
    printf("Grid size: [%d; %d]\n", gridSize, gridSize);
    printf("== Start training ==\n");

    mat4x4 m, p, mvp;
    float ratio;
    int width, height;

    glfwGetFramebufferSize(window, &width, &height);
    ratio = (float)width / (float)height;
    mat4x4_ortho(p, left, right, top, bottom, 1.f, -1.f);

    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_callback);
    printf("=== GRID ENVIRONMENT ===\n");
    printf("=== Controls: R-reset, E-toogle test/train mode LMB - set agent position ===\n");
    while (!glfwWindowShouldClose(window))
    {
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
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
        glColor3f(1.0, 0.0, 0.0);
        glLineWidth(1.2f);

        grid.Draw();

        agent.Discover();
        agent.Draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
	return 0;
}