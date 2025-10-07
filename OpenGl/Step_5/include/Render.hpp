// ============================================================================
//  FILENAME   : Render.hpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-19
//  UPDATED    : 2025-09-19
//  DESCRIPTION: Step 5 OpenGL - Interaction
// ============================================================================

#pragma once


#include "Shapes.hpp"
#include "Matrix4.hpp"
#include <algorithm>

unsigned int createShaderProgram();
void renderLoop(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram);

void handleKeybord(GLFWwindow* window, const float speed,
	std::vector<float>& offsetSq);
