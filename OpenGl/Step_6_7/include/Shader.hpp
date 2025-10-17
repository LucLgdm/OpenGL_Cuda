// ============================================================================
//  FILENAME   : Render.hpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-16
//  UPDATED    : 2025-10-17
//  DESCRIPTION: Step 6 OpenGL - Lighting
// ============================================================================

#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <glad/glad.h>

unsigned int createShader(const char* vertexPath, const char* fragmentPath);