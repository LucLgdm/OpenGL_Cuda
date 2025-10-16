// ============================================================================
//  FILENAME   : function.cuh
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-09
//  UPDATED    : 2025-10-16
//  DESCRIPTION: Step 5 Cuda - methods declaration
// ============================================================================

#pragma once

#include <iostream>
using namespace std;

#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16


// Reduction operations

struct Add {
	__device__ double operator()(double a, double b) const { return a + b; }
	__device__ double identity() const { return 0.0f; }
};

struct Max {
	__device__ double operator()(double a, double b) const { return fmaxf(a, b); }
	__device__ double identity() const { return -FLT_MAX; }
};

struct Mul {
	__device__ double operator()(double a, double b) const { return a * b; }
	__device__ double identity() const { return 1.0f; }
};

template<typename T, typename Op>
__global__ void reduceShared(const T* input, T* output, int size, Op op) {
	extern __shared__ T sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = (i < size) ? input[i] : op.identity();
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] = op(sdata[tid], sdata[tid + s]);
		}
		__syncthreads();
	}
	if (tid == 0) {
		output[blockIdx.x] = sdata[0];
	}
}

// Scan (Prefix Sum)

void scanLonelyBlock();
void scanMultiBlock();
__global__ void scanOneBlock(const int* input, int* output, int size);
__global__ void addOffsets(int* output, int* offsets, int size);
__global__ void scanBlock(const int* input, int* output, int* blockSums, int size);

// Histogramme avec shared memory

__global__ void histNaive(const unsigned int *input, int *histogram, int size, int BIN_COUNT);
__global__ void histShared(const unsigned int *input, int *histogram, int size, int BIN_COUNT);

__inline__ __device__ int laneId();
__inline__ __device__ int warpId();
__global__ void histWarpShared(const unsigned int *input, int *histogram, int size, int BIN_COUNT);

// Stencil / Convolution avancée

void multiConvolution(const int width, const int height);
void overlapping(const int width, const int height);
__global__ void convolutionGeneric(const float *input, float *output, int width, int height,
			const float *filter, int filterSize);
__global__ void threshold_kernel(const float* input, float* output, int width, int height, float threshold);

// Multi-kernel pipeline

__global__ void generateIntensity(float *input, int width, int height);

// Atomic operations et warp-synchronous programming

float maxCPU(const std::vector<float>& v);
__global__ void atomic_way(int *data);
__global__ void max_block(const float *input, float *maxBlock, int size);
__inline__ __device__ float warpReduceMax(float val);
__global__ void max_warp(const float* data, float* blockMax, int size);


// Simulation simple avec forces / interactions
// float2 : fourni par CUDA, pos.x et pos.y, make_float2(float x, float y)
struct Particle {
    float2 pos;
    float2 vel;
	float m;
};
__device__ inline float2 make_float2_device(float x, float y);
__global__ void updateParticles(Particle* particles, int n, float g, float dt, float eps);
