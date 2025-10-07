// ============================================================================
//  FILENAME   : Shapes.hpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-18
//  UPDATED    : 2025-09-18
//  DESCRIPTION: Step 4 OpenGL - Textures & UV Mapping
// ============================================================================

#pragma once

#include <iostream>
using namespace std;

#include <cmath>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
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

// -------------------------------------------------------------
// Shape type
// -------------------------------------------------------------
enum class ShapeType {
	TRIANGLE,
	SQUARE,
	CIRCLE
};

struct Triangle {
	Shape shape;
};

struct Square {
	Shape shape;
};

struct Circle {
	Shape shape;
};

Triangle createTriangle(string name, float size = 0.2f);
Square createSquare(string name, float size = 0.2f);
Circle createCircle(string name, Color col, float radius = 0.2f, int segments = 32);

void createShape(Shape &shape, float vertices[], size_t nver, unsigned int indices[] = nullptr, size_t nind = 0);
