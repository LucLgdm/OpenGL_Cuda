// ============================================================================
//  FILENAME   : Render.hpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-18
//  UPDATED    : 2025-09-19
//  DESCRIPTION: Step 4 OpenGL - Textures & UV Mapping
// ============================================================================

#pragma once


#include "Shapes.hpp"
#include "Matrix4.hpp"
#include <algorithm>

void renderShapes(const std::vector<Shape> &shapes, unsigned int shaderProgram);
unsigned int createShaderProgram();
void renderLoop(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram);

unsigned int createShaderProgram2();
void renderLoop2(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram);

unsigned int loadTexture(const char* path);