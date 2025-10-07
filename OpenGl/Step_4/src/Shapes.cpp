// ============================================================================
//  FILENAME   : Shapes.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-18
//  UPDATED    : 2025-09-18
//  DESCRIPTION: Step 4 OpenGL - Textures & UV Mapping
// ============================================================================

#include "Shapes.hpp"

Triangle createTriangle(string name, float size) {
	float vertices[] = {
		-size, -size, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, // rouge
		size,  -size, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, // vert
		0.0f,   size, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f // bleu
	};
	unsigned int indices[] = { 0, 1, 2 };

	Triangle tri;
	tri.shape._name = name;
	createShape(tri.shape, vertices, sizeof(vertices)/sizeof(float),
				indices, sizeof(indices)/sizeof(unsigned int));
	return tri;
}

Square createSquare(string name, float size) {
	float vertices[] = {
			// Pos		Color			tex coords
		-size, -size, 1.0f,0.0f,0.0f, 0.0f, 0.0f, // bottom left
		size,  -size, 0.0f,1.0f,0.0f, 1.0f, 0.0f, // bottom right
		size,   size, 0.0f,0.0f,1.0f, 1.0f, 1.0f, // top right
		-size,  size, 1.0f,1.0f,0.0f, 0.0f, 1.0f  // top left
	};
	unsigned int indices[] = { 0, 1, 2, 2, 3, 0 };

	Square sq;
	sq.shape._name = name;
	createShape(sq.shape, vertices, sizeof(vertices)/sizeof(float),
				indices, sizeof(indices)/sizeof(unsigned int));
	return sq;
}

Circle createCircle(string name, Color col, float radius, int segments) {
	std::vector<float> vertices;
	std::vector<unsigned int> indices;

	// centre
	vertices.push_back(0.0f); vertices.push_back(0.0f);
	vertices.push_back(0.8f); vertices.push_back(0.8f); vertices.push_back(0.8f);
	vertices.push_back(0.5f); vertices.push_back(0.5f);
	// vertices.push_back(col.r); vertices.push_back(col.g); vertices.push_back(col.b);

	// points sur le cercle
	for (int i = 0; i <= segments; i++) {
		float angle = 2.0f * M_PI * i / segments;
		float x = cos(angle) * radius;
		float y = sin(angle) * radius;
		vertices.push_back(x); vertices.push_back(y);
		vertices.push_back(col.r); vertices.push_back(col.g); vertices.push_back(col.b);
		
		float u = 0.5f + x / (2.0f * radius);
		float v = 0.5f + y / (2.0f * radius);
		vertices.push_back(u);
		vertices.push_back(v);
	}

	Circle c;
	c.shape._name = name;
	createShape(c.shape, vertices.data(), vertices.size(), nullptr, 0);
	c.shape.vertexCount = vertices.size() / 5; // important !
	c.shape.indexCount = 0;
	return c;
}


void createShape(Shape &shape, float vertices[], size_t nver,
				unsigned int indices[], size_t nind) {
	glGenVertexArrays(1, &shape.VAO);
	glGenBuffers(1, &shape.VBO);
	if (indices) glGenBuffers(1, &shape.EBO);

	glBindVertexArray(shape.VAO);

	glBindBuffer(GL_ARRAY_BUFFER, shape.VBO);
	glBufferData(GL_ARRAY_BUFFER, nver * sizeof(float), vertices, GL_STATIC_DRAW);

	if (indices) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, nind * sizeof(unsigned int), indices, GL_STATIC_DRAW);
		shape.indexCount = nind;
	} else {
		shape.vertexCount = nver / 7; // assuming 5 floats per vertex (pos+color)
	}

	// position attribute
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// texture attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(5 * sizeof(float)));
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);
}
