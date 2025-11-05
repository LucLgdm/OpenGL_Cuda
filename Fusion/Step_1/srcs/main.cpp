// ============================================================================
//  FILENAME   : main.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-17
//  UPDATED    : 2025-11-05
//  DESCRIPTION: Fusion : Step 1 - Ripple effect
// ============================================================================

#include <iostream>
using namespace std;

#include "ripple.hpp"

void framebuffer_size_callback(GLFWwindow* window) {
	glViewport(0, 0, width, height);
}

GLFWwindow* initWindow(const char* title) {
	if (!glfwInit()) return nullptr;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
	if (!window) { glfwTerminate(); return nullptr; }

	glfwMakeContextCurrent(window);

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "Failed to initialize GLAD\n";
		return nullptr;
	}
	return window;
}


void renderLoop(GLuint pbo, cudaGraphicsResource* cuda_pbo, GLuint tex, float time) {
    // Mapper le PBO pour CUDA
    uchar4* dptr; // Permet d'ecrire directement dans la memoire du pbo
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_pbo, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_pbo);

    // Lancer le kernel
    dim3 block(16,16);
    dim3 grid((width+15)/16, (height+15)/16);
    rippleKernel<<<grid, block>>>(dptr, width, height, time);

    // Démapper le PBO
    cudaGraphicsUnmapResources(1, &cuda_pbo, 0);

    // Mettre à jour la texture OpenGL
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
}

void drawQuadWithTexture(GLuint tex) {
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex);

    glBegin(GL_QUADS);
        glTexCoord2f(0,0); glVertex2f(-1,-1);
        glTexCoord2f(1,0); glVertex2f( 1,-1);
        glTexCoord2f(1,1); glVertex2f( 1, 1);
        glTexCoord2f(0,1); glVertex2f(-1, 1);
    glEnd();
}


int main() {
	GLFWwindow* window = initWindow("Ripple effect");
	if (!window) return -1;
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// On cree un PBO OpenGl : Buffer de memoire dans GPU
	GLuint pbo, tex;
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(GLubyte) * 4, nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// On enregistre le pbo sur cuda
	cudaGraphicsResource* cuda_pbo;
	cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

	while (!glfwWindowShouldClose(window)) {
		float time = glfwGetTime();
		renderLoop(pbo, cuda_pbo, tex, time);

		// Rendu quad plein écran
		drawQuadWithTexture(tex);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	cudaGraphicsUnregisterResource(cuda_pbo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glfwTerminate();

	return 0;
}
