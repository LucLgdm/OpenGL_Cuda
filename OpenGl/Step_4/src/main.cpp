// ============================================================================
//  FILENAME   : main.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-18
//  UPDATED    : 2025-09-18
//  DESCRIPTION: Step 4 OpenGL - Textures & UV Mapping
// ============================================================================

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <iostream>
using namespace std;

#include "Shapes.hpp"
#include "Render.hpp"

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
	GLFWwindow* window = initWindow(800, 800, "Shapes");
	if (!window) return -1;
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


	unsigned int shaderProgram = createShaderProgram();
	vector<Shape> shapes;
	vector<Color> colors;

	colors.push_back({1.0f, 0.0f, 0.0f}); 
	colors.push_back({0.0f, 1.0f, 0.0f});
	colors.push_back({0.0f, 0.0f, 1.0f});
	
	Triangle tri = createTriangle("triangle", 0.3f);
	Square sq   = createSquare("carre", 0.4f);
	Circle earth = createCircle("earth", colors[2], 0.2f);
	Circle moon = createCircle("moon", colors[1], 0.2f);
	Circle moon2 = createCircle("moon2", colors[0], 0.2f);
	shapes.push_back(sq.shape);
	shapes.push_back(earth.shape); shapes.push_back(moon.shape); shapes.push_back(moon2.shape);
	// shapes.push_back(tri.shape);
	renderLoop(window, shapes, shaderProgram);

	cleanup(shapes, shaderProgram, window);
	return 0;
}
