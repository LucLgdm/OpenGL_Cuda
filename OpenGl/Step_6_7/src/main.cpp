// ============================================================================
//  FILENAME   : main.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-16
//  UPDATED    : 2025-10-17
//  DESCRIPTION: Step 6 OpenGL - Lighting
// ============================================================================


#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <iostream>
using namespace std;

#include "Shapes.hpp"
#include "Render.hpp"
#include "Shader.hpp"

// -------------------------------------------------------------
// Callbacks
// -------------------------------------------------------------
/**********************************************************
 * Set the viewport to cover the entire window
 **********************************************************/
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

// -------------------------------------------------------------
// Initialization
// -------------------------------------------------------------
GLFWwindow* initWindow(int width, int height, const char* title)
{
	if (!glfwInit()) return nullptr;
	/**********************************************************
	 * Configure the window and OpenGL context:
	 *  - OpenGL version 3.3
	 *  - Use the Core Profile
	 * Must be done before glfwCreateWindow()
	 **********************************************************/
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
	if (!window) { glfwTerminate(); return nullptr; }

	/***********************************************************
	 * Make this OpenGL context current for all OpenGL commands
	 ***********************************************************/
	glfwMakeContextCurrent(window);

	/***********************************************************
	 * Automatically handle window resizing
	 ***********************************************************/
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "Failed to initialize GLAD\n";
		return nullptr;
	}
	return window;
}

// -------------------------------------------------------------
// Shaders
// -------------------------------------------------------------
void cleanup(const std::vector<Shape> &shapes, unsigned int shaderProgram, GLFWwindow* window)
{
	for (const auto &shape : shapes) {
		glDeleteVertexArrays(1, &shape.VAO);
		if (shape.indexCount)
			glDeleteBuffers(1, &shape.EBO);
		glDeleteBuffers(1, &shape.VBO);
	}
	glDeleteProgram(shaderProgram);
	glfwDestroyWindow(window);
	glfwTerminate();
}

// -------------------------------------------------------------
// Main
// -------------------------------------------------------------
int main() {
	GLFWwindow* window = initWindow(900, 900, "Shapes");
	if (!window) return -1;
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	unsigned int shaderProgram = createShader("./src/vertex_shader.glsl", "./src/fragment_shader.glsl");
	// unsigned int shaderProgram = createShader("./src/vertex_shader.glsl", "./src/fragment_shader_multiLight.glsl");
	// unsigned int shaderProgram = createShader("./src/vertex_shader.glsl", "./src/fragment_shader_spotlight.glsl");

	vector<Shape> shapes;

	// Color color = {static_cast<float>(rand()) / RAND_MAX, 0.5f, static_cast<float>(rand()) / RAND_MAX};
	
	Shape cube = createCube(1.0f);
	Shape sphere = createSphere(2.0f, 45, 45);
	Shape cube2 = createCube(1.0f);
	Shape orbite = createSphere(1.0f, 60, 60);
	cube2._name = "cube2";
	orbite._name = "orbite";
	shapes.push_back(cube);
	shapes.push_back(sphere);
	shapes.push_back(cube2);
	shapes.push_back(orbite);


	renderLoop(window, shapes, shaderProgram);

	cleanup(shapes, shaderProgram, window);
	return 0;
}
