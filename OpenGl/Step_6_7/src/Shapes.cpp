// ============================================================================
//  FILENAME   : Shapes.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-16
//  UPDATED    : 2025-10-17
//  DESCRIPTION: Step 6 OpenGL - Lighting
// ============================================================================

#include "Shapes.hpp"

Shape createCube(float size) {
	float hs = size / 2.0f;

	// vertices : position(x,y,z) + couleur(R, G, B) + normale(nx,ny,nz)
	std::vector<float> vertices = {
		// Face +Z (rouge)
		-hs,-hs, hs,         1, 0, 0,          0, 0, 1,
		 hs,-hs, hs,         1, 0, 0,          0, 0, 1,
		 hs, hs, hs,         1, 0, 0,          0, 0, 1,
		-hs, hs, hs,         1, 0, 0,          0, 0, 1,

		// Face -Z (vert)
		-hs,-hs,-hs,         0, 1, 0,          0, 0, -1,
		 hs,-hs,-hs,         0, 1, 0,          0, 0, -1,
		 hs, hs,-hs,         0, 1, 0,          0, 0, -1,
		-hs, hs,-hs,         0, 1, 0,          0, 0, -1,

		// Face +X (bleu)
		hs,-hs,-hs,          0, 0, 1,          1, 0, 0,
		hs,-hs, hs,          0, 0, 1,          1, 0, 0,
		hs, hs, hs,          0, 0, 1,          1, 0, 0,
		hs, hs,-hs,          0, 0, 1,          1, 0, 0,

		// Face -X (jaune)
		-hs,-hs,-hs,         1, 1, 0,         -1, 0, 0,
		-hs,-hs, hs,         1, 1, 0,         -1, 0, 0,
		-hs, hs, hs,         1, 1, 0,         -1, 0, 0,
		-hs, hs,-hs,         1, 1, 0,         -1, 0, 0,

		// Face +Y (cyan)
		-hs, hs,-hs,         0, 1, 1,          0, 1, 0,
		 hs, hs,-hs,         0, 1, 1,          0, 1, 0,
		 hs, hs, hs,         0, 1, 1,          0, 1, 0,
		-hs, hs, hs,         0, 1, 1,          0, 1, 0,

		// Face -Y (magenta)
		-hs,-hs,-hs,         1, 0, 1,          0, -1, 0,
		 hs,-hs,-hs,         1, 0, 1,          0, -1, 0,
		 hs,-hs, hs,         1, 0, 1,          0, -1, 0,
		-hs,-hs, hs,         1, 0, 1,          0, -1, 0
	};

	// indices pour 12 triangles (2 par face)
	std::vector<unsigned int> indices = {
		0 , 1 , 2 , 0 , 2 ,  3, // front
		4 , 5 , 6 , 4 , 6 ,  7, // back
		8 , 9 , 10, 8 , 10, 11, // left
		12, 13, 14, 12, 14, 15, // right
		16, 17, 18, 16, 18, 19, // top
		20, 21, 22, 20, 22, 23  // bottom
	};

	return createShape("cube", vertices, indices);
}


Shape createSphere(float radius, unsigned int sectorCount, unsigned int stackCount) {
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    for(unsigned int i = 0; i <= stackCount; ++i) {
        float stackAngle = glm::pi<float>() / 2 - i * glm::pi<float>() / stackCount; // de pi/2 à -pi/2
        float xy = radius * cosf(stackAngle);
        float z = radius * sinf(stackAngle);

        for(unsigned int j = 0; j <= sectorCount; ++j) {
            float sectorAngle = j * 2 * glm::pi<float>() / sectorCount; // de 0 à 2pi

            float x = xy * cosf(sectorAngle);
            float y = xy * sinf(sectorAngle);
            vertices.push_back(x);  // position
            vertices.push_back(y);
            vertices.push_back(z);

            // couleur : simple gradient selon z
            float r = (z / radius + 1.0f) * 0.5f;
            float g = 0.5f;
            float b = 1.0f - r;
            vertices.push_back(r);
            vertices.push_back(g);
            vertices.push_back(b);

            // normales
            glm::vec3 norm = glm::normalize(glm::vec3(x, y, z));
            vertices.push_back(norm.x);
            vertices.push_back(norm.y);
            vertices.push_back(norm.z);
        }
    }

    // indices
    for(unsigned int i = 0; i < stackCount; ++i) {
        for(unsigned int j = 0; j < sectorCount; ++j) {
            unsigned int first = i * (sectorCount + 1) + j;
            unsigned int second = first + sectorCount + 1;

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }

    return createShape("sphere", vertices, indices);
}



Shape createShape(const std::string& name, const std::vector<float>& vertices, const std::vector<unsigned int>& indices) {
	Shape shape;
	shape._name = name;
	shape.vertexCount = vertices.size() / 9; // 3 pos + 3 couleurs + 3 normales
	shape.indexCount = indices.size();

	glGenVertexArrays(1, &shape.VAO);
	glGenBuffers(1, &shape.VBO);
	glGenBuffers(1, &shape.EBO);

	glBindVertexArray(shape.VAO);

	glBindBuffer(GL_ARRAY_BUFFER, shape.VBO);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

	// positions
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	// couleurs
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// normales
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);

	return shape;
}
