// ============================================================================
//  FILENAME   : main.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-18
//  UPDATED    : 2025-09-18
//  DESCRIPTION: Step 3 OpenGL - Transformations: translation, rotation, scaling
// ============================================================================

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <iostream>
using namespace std;


#include "matrix4.hpp"

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

unsigned int createShaderProgram()
{
	const char* vertexShaderSource = R"(
		#version 330 core
		layout(location = 0) in vec2 aPos;
		layout(location = 1) in vec3 aColor;

		out vec3 vertexColor;
		uniform mat4 transform;  // Transformation matrix

		void main() {
			gl_Position = transform * vec4(aPos, 0.0, 1.0);
			// gl_Position = vec4(aPos, 0.0, 1.0);
			vertexColor = aColor;
		}
		)";

    const char* fragmentShaderSource = R"(
		#version 330 core
		in vec3 vertexColor;  // from vertex shader
		out vec4 FragColor;

		void main() {
			FragColor = vec4(vertexColor, 1.0);
		}
    )";

	/**********************************************************
	 * Writes the shader code and compiles it
	 **********************************************************/
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

	/**********************************************************
	 * Creates a shader program, attach shaders and link them
	 **********************************************************/
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

// -------------------------------------------------------------
// Shape type
// -------------------------------------------------------------
enum class ShapeType {
    TRIANGLE,
    SQUARE,
    CIRCLE
};

// -------------------------------------------------------------
// Generate vertices
// Each vertex: x, y, r, g, b
// -------------------------------------------------------------
void generateVertices(std::vector<float> &vertices, std::vector<unsigned int> &indices, ShapeType type,
					float size = 0.2f, int circleSegments = 32) {
	vertices.clear();
	indices.clear();

	switch(type) {
		case ShapeType::TRIANGLE:
			vertices = {
				// -size, -size, 1.0f, 0.0f, 0.0f,
				// size, -size, 0.0f, 1.0f, 0.0f,
				// 0.0f, size,  0.0f, 0.0f, 1.0f
				-0.8f, 0.6f, 1.0f, 0.0f, 0.0f,
				-0.6f, 0.6f, 0.0f, 1.0f, 0.0f,
				-0.7f, 0.95f, 0.0f, 0.0f, 1.0f
			};
			indices = { 0, 1, 2 };
			break;

		case ShapeType::SQUARE:
			vertices = {
				-size, -size, 1.0f, 0.0f, 0.0f,
				size, -size, 0.0f, 1.0f, 0.0f,
				size,  size, 0.0f, 0.0f, 1.0f,
				-size,  size, 1.0f, 1.0f, 0.0f
				// -0.2f, -0.2f, 1.0f, 0.0f, 0.0f,
				// 0.2f , -0.2f, 0.0f, 1.0f, 0.0f,
				// 0.2f , 0.2f , 0.0f, 0.0f, 1.0f,
				// -0.2f, 0.2f , 1.0f, 1.0f, 0.0f
			};
			indices = { 0, 1, 2, 2, 3, 0 };
			break;

		case ShapeType::CIRCLE:
			float centerX = 0.0f, centerY = 0.0f;
			vertices.push_back(centerX); vertices.push_back(centerY); // center
			vertices.push_back(1.0f); vertices.push_back(0.38f); vertices.push_back(1.0f); // white

			for(int i = 0; i <= circleSegments; ++i) {
				float angle = 2.0f * M_PI * i / circleSegments;
				float x = cos(angle) * size + centerX;
				float y = sin(angle) * size + centerY;
				vertices.push_back(x); vertices.push_back(y);
				vertices.push_back(1.0f); vertices.push_back(0.38f); vertices.push_back(0.74f); // perimeter
			}
			// no indices for GL_TRIANGLE_FAN
			break;
	}
}


struct Shape {
    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO; // optional, for indexed shapes
    size_t indexCount; // number of indices (for glDrawElements)
    size_t vertexCount; // number of vertices (for glDrawArrays)
};

// -------------------------------------------------------------
// Initialize a shape
// -------------------------------------------------------------
void createShape(Shape &shape, float vertices[], size_t nver,
				unsigned int indices[] = nullptr, size_t nind = 0)
{
	glGenVertexArrays(1, &shape.VAO);
	glGenBuffers(1, &shape.VBO);
	if (indices) glGenBuffers(1, &shape.EBO);

	glBindVertexArray(shape.VAO);

	glBindBuffer(GL_ARRAY_BUFFER, shape.VBO);
	glBufferData(GL_ARRAY_BUFFER, nver * sizeof(float), vertices, GL_STATIC_DRAW);

	if (indices) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, nind * sizeof(unsigned int), indices, GL_STATIC_DRAW);
		shape.indexCount = nind;
	} else {
		shape.vertexCount = nver / 5; // assuming 5 floats per vertex (pos+color)
	}

	// position attribute
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
}

// -------------------------------------------------------------
// Render multiple shapes
// -------------------------------------------------------------
void renderShapes(const std::vector<Shape> &shapes, unsigned int shaderProgram)
{
    glUseProgram(shaderProgram);
    for (const auto &shape : shapes) {
        glBindVertexArray(shape.VAO);
        if (shape.indexCount)
            glDrawElements(GL_TRIANGLES, shape.indexCount, GL_UNSIGNED_INT, 0);
        else
            glDrawArrays(GL_TRIANGLE_FAN, 0, shape.vertexCount);
    }
}


// -------------------------------------------------------------
// Render
// -------------------------------------------------------------
void renderShape(unsigned int VAO, ShapeType type, size_t vertexCount) {
	glBindVertexArray(VAO);

	if (type == ShapeType::TRIANGLE)
		glDrawArrays(GL_TRIANGLES, 0, vertexCount);
	else if (type == ShapeType::SQUARE)
		glDrawArrays(GL_TRIANGLE_FAN, 0, vertexCount); // easier than indices for 4 vertices
	else if (type == ShapeType::CIRCLE)
		glDrawArrays(GL_TRIANGLE_FAN, 0, vertexCount);

	glBindVertexArray(0);
}

// -------------------------------------------------------------
// Render loop for multiple shapes
// -------------------------------------------------------------
void renderLoop(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram) {
	float offset = 0.4f;
	int direction = 1;
	while (!glfwWindowShouldClose(window)) {
		glClearColor(0.4f, 0.3f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(shaderProgram);

		// Draw shapes
		// for (const auto &shape : shapes) {
		for(size_t i = 0; i < shapes.size(); i++) {
			glBindVertexArray(shapes[i].VAO);

			Mat4 transform = identity();
			if (i == 1) {
				offset += 0.001f * direction;
				if (offset >= 1.0f) {
					offset = 1.0f;
					direction = -1;
				} else if (offset <= 0.4f) {
					offset = 0.4f;
					direction = 1;
				}
				
				float angle = (float)glfwGetTime();
				Mat4 s = scale(0.25f, 0.25f, 1.0f);
				Mat4 t = translate(offset, 0.0f, 0.0f); // éloignement du centre
				Mat4 r = rotateZ(angle); // rotation autour du centre
				transform = multiply(s, multiply(t, r)); // rotation autour du centre, on se decale puis on tourne
				// transform = multiply(r, transform); // rotation sur lui-meme et autour du centre
			}
			unsigned int transformLoc = glGetUniformLocation(shaderProgram, "transform");
			glUniformMatrix4fv(transformLoc, 1, GL_FALSE, transform.m);
			
			if (shapes[i].indexCount)
				glDrawElements(GL_TRIANGLES, shapes[i].indexCount, GL_UNSIGNED_INT, 0);
			else
				glDrawArrays(GL_TRIANGLE_FAN, 0, shapes[i].vertexCount);
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

}

// -------------------------------------------------------------
// Cleanup
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

	unsigned int shaderProgram = createShaderProgram();

	std::vector<Shape> shapes;

	// Triangle
	// std::vector<float> triVertices;
	// std::vector<unsigned int> triIndices;
	// Shape triangle;
	// generateVertices(triVertices, triIndices, ShapeType::TRIANGLE);
	// createShape(triangle, triVertices.data(), triVertices.size(),
	// 			triIndices.data(), triIndices.size());
	// shapes.push_back(triangle);

	// // Square
	// std::vector<float> sqVertices;
	// std::vector<unsigned int> sqIndices;
	// Shape square;
	// generateVertices(sqVertices, sqIndices, ShapeType::SQUARE);
	// createShape(square, sqVertices.data(), sqVertices.size(),
	// 			sqIndices.data(), sqIndices.size());
	// shapes.push_back(square);

	// Circle
	std::vector<float> circVertices;
	std::vector<unsigned int> circIndices; // empty
	Shape circle;
	generateVertices(circVertices, circIndices, ShapeType::CIRCLE);
	createShape(circle, circVertices.data(), circVertices.size(),
				circIndices.data(), circIndices.size()); // indices.size() == 0
	circle.vertexCount = circVertices.size() / 5; // important !
	circle.indexCount = 0;
	shapes.push_back(circle);

	std::vector<float> circVertices2;
	std::vector<unsigned int> circIndices2; // empty
	Shape circle2;
	generateVertices(circVertices2, circIndices2, ShapeType::CIRCLE);
	createShape(circle2, circVertices2.data(), circVertices2.size(),
				circIndices2.data(), circIndices2.size()); // indices.size() == 0
	circle2.vertexCount = circVertices2.size() / 5; // important !
	circle2.indexCount = 0;
	shapes.push_back(circle2);

	renderLoop(window, shapes, shaderProgram);
	// renderLoop(window, VAO, ShapeType::CIRCLE, vertices.size() / 5, shaderProgram);

	cleanup(shapes, shaderProgram, window);
	return 0;
}
