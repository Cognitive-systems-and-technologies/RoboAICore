#include <stdio.h>
#include <stdlib.h>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "linmath.h"
#include <math.h>

#include "cmd/cartpole_cont/cart.h"
#include "box2d/box2d.h"

#include "TCommon.h"
#include "agent.h"

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
	glfwSwapInterval(1.0);

	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_MULTISAMPLE);  // Enabled Multisample 
	//glEnable(GL_LINE_WIDTH);
	//glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBlendEquation(GL_FUNC_ADD);
	return window;
}

Cart* cartLink;
Agent* agentLink;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_R && action == GLFW_PRESS) {
		cartLink->Reset();
	}
	if (key == GLFW_KEY_W && action == GLFW_PRESS) {
		
	}
	if (key == GLFW_KEY_S && action == GLFW_PRESS) {
		
	}
	if (key == GLFW_KEY_A && action == GLFW_PRESS) {
		//cartLink->pushLeft();
		printf("Push the cart left\n");
		cartLink->b2->m_body->ApplyLinearImpulseToCenter(b2Vec2(-0.2f, 0), true);
	}
	if (key == GLFW_KEY_D && action == GLFW_PRESS) {
		//cartLink->pushRight();
		printf("Push the cart right\n");
		cartLink->b2->m_body->ApplyLinearImpulseToCenter(b2Vec2(0.2f, 0), true);
	}
	if (key == GLFW_KEY_F && action == GLFW_PRESS) {

	}
	if (key == GLFW_KEY_E && action == GLFW_PRESS) {
		printf("E pressed\n");
		agentLink->phase = agentLink->phase == Agent::Phase::TRAIN ? Agent::Phase::TEST : Agent::Phase::TRAIN;
		if (agentLink->phase == Agent::Phase::TEST) {
			glfwSwapInterval(1);
			printf("TEST PHASE\n");
		}
		else {
			glfwSwapInterval(0.5);
			printf("TRAIN PHASE\n");
		}
	}
}

int main() 
{
	b2Vec2 gravity(0.0f, -9.8f);
	b2World world(gravity);
	
	int width = 640, height = 480;
	GLFWwindow* window = CreateGLFWindow(width, height);
	//int width, height;
	float aspect = (float)width / (float)height;
	float viewRectSize = 1.f;
	float right = viewRectSize * aspect, top = viewRectSize;
	float bottom = -viewRectSize, left = -viewRectSize * aspect;
	glClearColor(1.f, 1.f, 1.f, 0.0f);

	mat4x4 m, p, mvp;
	mat4x4_ortho(p, left, right, bottom, top, 1.f, -1.f);
	
	Cart cart(&world);
	cartLink = &cart;

	Agent agent(&cart);
	agentLink = &agent;

	glfwSetKeyCallback(window, key_callback);

	float lastFrame = 0;
	float deltaTime = 0;
	
	float time = 0;
	//sudo apt-get install xorg-dev
	printf("=== CARTPOLE ENVIRONMENT ===\n");
	printf("=== Controls: R-reset, E-toogle test/train mode ===\n");
	printf("=== Controls: A-push the cart left, D-push the cart right ===\n");
	while (!glfwWindowShouldClose(window))
	{
		float currentFrame = (float)glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		time += deltaTime;

		glViewport(0, 0, width, height);
		glClear(GL_COLOR_BUFFER_BIT);
		glLoadMatrixf((const GLfloat*)p);

		agent.Discover();
		cart.Draw();

		//printf("pole angle: %f\n", cart.poleAngle());
		//printf("cart pos x:%f\n", cart.cartPos());
		//printf("cart velocity: %f\n", cart.cartVel());
		//printf("pole velocity: %f\n", cart.poleVel());
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}