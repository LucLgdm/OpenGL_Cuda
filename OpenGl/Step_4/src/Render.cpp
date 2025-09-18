// ============================================================================
//  FILENAME   : Render.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-18
//  UPDATED    : 2025-09-18
//  DESCRIPTION: Step 4 OpenGL - Textures & UV Mapping
// ============================================================================

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Render.hpp"

void renderShapes(const std::vector<Shape> &shapes, unsigned int shaderProgram) {
	glUseProgram(shaderProgram);
	for (const auto &shape : shapes) {
		glBindVertexArray(shape.VAO);
		if (shape.indexCount)
			glDrawElements(GL_TRIANGLES, shape.indexCount, GL_UNSIGNED_INT, 0);
		else
			glDrawArrays(GL_TRIANGLE_FAN, 0, shape.vertexCount);
	}
}


void renderLoop(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram) {
	unsigned int texture = loadTexture("image.png"); // à faire une seule fois
	glUseProgram(shaderProgram);
	glUniform1i(glGetUniformLocation(shaderProgram, "ourTexture"), 0);

	while (!glfwWindowShouldClose(window)) {
		glClearColor(0.4f, 0.3f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// Draw shapes
		for (const auto &shape : shapes) {
			glBindVertexArray(shape.VAO);

			Mat4 transform = identity();
			float angle;
			Mat4 t, s, r;
			if (shape._name == "carre") {
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, texture);
				glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1);
				t = translate(0.5f, 0.0f, 0.0f);
				s = scale(0.3f, 0.3f, 0.0f);
				r = rotateZ((float)glfwGetTime());
				transform = multiply(s, t);
				transform = multiply(r, transform);
			}else{
				glBindTexture(GL_TEXTURE_2D, 0); // Desactive la texture
				glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0);
			}

			if (shape._name == "moon" || shape._name == "moon2") {	
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, texture);
				glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 1);	
				if (shape._name == "moon"){
					t = translate(0.4f, 0.0f, 0.0f); // éloignement du centre
					angle = (float)glfwGetTime();
					s = scale(0.25f, 0.25f, 1.0f);
				}else{
					t = translate(0.6f, 0.0f, 0.0f); // éloignement du centre
					angle =  0.24 * (float)glfwGetTime();
					s = scale(0.50f, 0.50f, 1.0f);
				}
				r = rotateZ(angle); // rotation autour du centre
				transform = multiply(s, multiply(t, r)); // rotation autour du centre, on se decale puis on tourne
				// transform = multiply(r, transform); // rotation sur lui-meme et autour du centre
			}

			unsigned int transformLoc = glGetUniformLocation(shaderProgram, "transform");
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
		layout(location = 0) in vec2 aPos;
		layout(location = 1) in vec3 aColor;
		layout(location = 2) in vec2 aTexCoord;

		out vec3 vertexColor;
		out vec2 texCoord;
		uniform mat4 transform;  // Transformation matrix

		void main() {
			gl_Position = transform * vec4(aPos, 0.0, 1.0);
			vertexColor = aColor;
			texCoord = aTexCoord;
		}
		)";

    const char* fragmentShaderSource = R"(
		#version 330 core
		in vec3 vertexColor;  // from vertex shader
		in vec2 texCoord;
		out vec4 FragColor;

		uniform sampler2D ourTexture;
		uniform bool useTexture;

		void main() {
			vec4 baseColor = vec4(vertexColor, 1.0);
			if (useTexture)
				baseColor = texture(ourTexture, texCoord); // * baseColor; Pour avoir la couleur en plus...
			FragColor = baseColor;
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


unsigned int loadTexture(const char* path) {
	unsigned int textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Paramètres de la texture
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	int width, height, nrChannels;
	unsigned char *data = stbi_load(path, &width, &height, &nrChannels, 4);
	if (data) {
		GLenum format = (nrChannels == 3) ? GL_RGB : GL_RGBA;
		glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	} else {
		std::cout << "Failed to load texture: " << path << std::endl;
	}
	stbi_image_free(data);
	return textureID;
}