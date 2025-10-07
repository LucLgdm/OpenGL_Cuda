// ============================================================================
//  FILENAME   : matrix4.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-18
//  UPDATED    : 2025-09-18
//  DESCRIPTION: Step 3 OpenGL - Transformations: translation, rotation, scaling
// ============================================================================

#include "Matrix4.hpp"

Mat4 identity() {
	Mat4 mat = { {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	}};
	return mat;
}


Mat4 multiply(const Mat4 &a, const Mat4 &b) {
	Mat4 result = {0};
	for (int row = 0; row < 4; row++) {
		for (int col = 0; col < 4; col++) {
			for (int k = 0; k < 4; k++) {
				result.m[row*4 + col] += a.m[row*4 + k] * b.m[k*4 + col];
			}
		}
	}
	return result;
}

Mat4 translate(float x, float y, float z) {
	Mat4 mat = identity();
	mat.m[12] = x;
	mat.m[13] = y;
	mat.m[14] = z;
	return mat;
}

Mat4 scale(float sx, float sy, float sz) {
	Mat4 mat = identity();
	mat.m[0] = sx;
	mat.m[5] = sy;
	mat.m[10] = sz;
	return mat;
}

Mat4 rotateZ(float angle) {
	Mat4 mat = identity();
	float c = cos(angle);
	float s = sin(angle);
	mat.m[0] = c;  mat.m[4] = -s;
	mat.m[1] = s;  mat.m[5] =  c;
	return mat;
}
