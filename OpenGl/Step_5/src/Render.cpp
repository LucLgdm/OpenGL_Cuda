// ============================================================================
//  FILENAME   : Render.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-19
//  UPDATED    : 2025-09-19
//  DESCRIPTION: Step 5 OpenGL - Interaction
// ============================================================================

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Render.hpp"

void renderLoop(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram) {
	glUseProgram(shaderProgram);

	float speed = 0.005f;
	vector<float> offsetSq(2, 0.0f);

	while (!glfwWindowShouldClose(window)) {
		glClearColor(0.3f, 0.2f, 0.23f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		handleKeybord(window, speed, offsetSq);

		unsigned int transformLoc = glGetUniformLocation(shaderProgram, "transform");
		
		Mat4 transform = identity();
		Mat4 r = rotateZ((float)glfwGetTime());
		for (const auto &shape : shapes) {
			glBindVertexArray(shape.VAO);

			if (shape._name == "square") {
                transform = translate(offsetSq[0], offsetSq[1], 0.0f);
				transform = multiply(r, transform);
            } else if (shape._name == "circle") {
                double xpos, ypos;
				glfwGetCursorPos(window, &xpos, &ypos);

				int width, height;
				glfwGetWindowSize(window, &width, &height);

				// conversion en coordonnées NDC
				float nx = (xpos / width) * 2.0f - 1.0f;
				float ny = 1.0f - (ypos / height) * 2.0f;

				transform = translate(nx, ny, 0.0f);
            }

			glUniformMatrix4fv(transformLoc, 1, GL_FALSE, transform.m);

			if (shape.indexCount)
				glDrawElements(GL_TRIANGLES, shape.indexCount, GL_UNSIGNED_INT, 0);
			else
				glDrawArrays(GL_TRIANGLE_FAN, 0, shape.vertexCount);
		}
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

unsigned int createShaderProgram() {
	const char* vertexShaderSource = R"(
		#version 330 core
		layout (location = 0) in vec2 aPos;      // 2 floats
		layout (location = 1) in vec3 aColor;    // 3 floats

		out vec3 vertexColor;
		uniform mat4 transform;

		void main() {
			gl_Position = transform * vec4(aPos, 0.0, 1.0);
			vertexColor = aColor;
		}
	)";

	const char* fragmentShaderSource = R"(
		#version 330 core
		in vec3 vertexColor;
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

void handleKeybord(GLFWwindow* window, const float speed, std::vector<float>& offsetSq) {
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		offsetSq[1] += speed;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		offsetSq[1] -= speed;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		offsetSq[0] -= speed;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		offsetSq[0] += speed;

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}
