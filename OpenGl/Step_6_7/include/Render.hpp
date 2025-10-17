// ============================================================================
//  FILENAME   : Render.hpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-16
//  UPDATED    : 2025-10-17
//  DESCRIPTION: Step 6 OpenGL - Lighting
// ============================================================================

#pragma once


#include "Shapes.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>

unsigned int createShaderProgram();
void renderLoop(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram);
