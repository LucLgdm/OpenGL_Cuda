// ============================================================================
//  FILENAME   : ripple.hpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-28
//  UPDATED    : 2025-11-05
//  DESCRIPTION: Fusion : Step 1 - Ripple effect
// ============================================================================

#pragma once

#include <iostream>
using namespace std;

#include "kernel.cuh"
#include <bits/stdc++.h>
#include "../../external/glad/glad.h"
#include <GLFW/glfw3.h>

const int width = 900;
const int height = 900;

GLFWwindow* initWindow(int width, int height, const char* title);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void drawQuadWithTexture(GLuint tex);
void renderLoop(GLuint pbo, cudaGraphicsResource* cuda_pbo, GLuint tex, float time);



