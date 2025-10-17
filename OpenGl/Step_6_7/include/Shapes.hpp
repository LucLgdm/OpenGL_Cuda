// ============================================================================
//  FILENAME   : Shapes.hpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-19
//  UPDATED    : 2025-10-17
//  DESCRIPTION: Step 5 OpenGL - Interaction
// ============================================================================

#pragma once

#include <iostream>
using namespace std;

#include <cmath>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <string>


struct Color {
	float r;
	float g;
	float b;
};

struct Shape {
	string _name;
	unsigned int VAO;
	unsigned int VBO;
	unsigned int EBO; // optional, for indexed shapes
	size_t indexCount; // number of indices (for glDrawElements)
	size_t vertexCount; // number of vertices (for glDrawArrays)
};

Shape createCube(float size);

Shape createShape(const std::string& name, const std::vector<float>& vertices, const std::vector<unsigned int>& indices);
Shape createSphere(float radius, unsigned int sectorCount, unsigned int stackCount);
