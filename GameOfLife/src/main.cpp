/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: lde-merc <lde-merc@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/09/10 16:06:00 by lde-merc          #+#    #+#             */
/*   Updated: 2025/09/11 13:35:09 by lde-merc         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "World.hpp"

unsigned int createShaderProgram(const char* vertexSrc, const char* fragmentSrc) {
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSrc, nullptr);
    glCompileShader(vertexShader);

    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "Vertex Shader compilation failed:\n" << infoLog << std::endl;
    }

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSrc, nullptr);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "Fragment Shader compilation failed:\n" << infoLog << std::endl;
    }

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader linking failed:\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}


int main() {
	cout << "Game of Life" << endl;

	if (!glfwInit()) return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(1600, 1200, "Game of Life", nullptr, nullptr);
	if (!window) {
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// Load OpenGL functions with GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "Failed to initialize GLAD\n";
		return -1;
	}

	glEnable(GL_PROGRAM_POINT_SIZE); // allow custom point size


	const char* vertexShaderSource = R"(
	#version 330 core
	layout(location = 0) in vec2 aPos;
	uniform float uScale;
	uniform vec2 uOffset;
	void main() {
		vec2 pos = (aPos + uOffset) * uScale;
		gl_Position = vec4(pos, 0.0, 1.0);
		gl_PointSize = 7.0;
	}
	)";

	const char* fragmentShaderSource = R"(
	#version 330 core
	out vec4 FragColor;
	void main() {
		FragColor = vec4(1.0); // white
	}
	)";
	unsigned int shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
	
	World world;
	
	unsigned int VAO, VBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW); // dynamic buffer

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);


	float scale = 0.02f;       // size of each cell in NDC
	float offsetX = 0.0f;
	float offsetY = 0.0f;
	float lastTime = 0.0f;
	float stepInterval = 0.05f;


	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);

		// Step simulation every frame (or every N frames)
		
		float currentTime = glfwGetTime();
		if (currentTime - lastTime >= stepInterval) {
			world.step();        // update simulation
			lastTime = currentTime;
		}

		// Convert alive cells to vertices
		std::vector<float> vertices;
		for (auto& cell : world.grid().getAliveCells()) {
			vertices.push_back(static_cast<float>(cell.first));
			vertices.push_back(static_cast<float>(cell.second));
		}

		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);

		glUseProgram(shaderProgram);
		glUniform1f(glGetUniformLocation(shaderProgram, "uScale"), scale);
		glUniform2f(glGetUniformLocation(shaderProgram, "uOffset"), offsetX, offsetY);

		glBindVertexArray(VAO);
		glDrawArrays(GL_POINTS, 0, vertices.size() / 2);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	return 0;
}