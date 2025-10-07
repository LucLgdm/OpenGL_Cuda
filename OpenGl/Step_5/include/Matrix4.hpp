// ============================================================================
//  FILENAME   : matrix4.hpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-19
//  UPDATED    : 2025-09-19
//  DESCRIPTION: Step 5 OpenGL - Interaction
// ============================================================================

#include <iostream>
#include <cmath>

struct Mat4 {
	float m[16];
};

Mat4 identity();
Mat4 multiply(const Mat4 &a, const Mat4 &b);
Mat4 translate(float x, float y, float z);
Mat4 scale(float sx, float sy, float sz);
Mat4 rotateZ(float angle);
