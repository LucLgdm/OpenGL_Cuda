// ============================================================================
//  FILENAME   : function.cuh
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-09
//  UPDATED    : 2025-10-09
//  DESCRIPTION: Step 5 Cuda - methods declaration
// ============================================================================

#pragma once

#include <iostream>
using namespace std;

#include <bits/stdc++.h>
#include <cuda_runtime.h>

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

void advancedReduction();
template<typename T, typename Op>
__global__ void reduceShared(const T* input, T* output, int size, Op op);

// Scan (Prefix Sum)

void scan();
void scanLonelyBlock();
void scanMultiBlock();
__global__ void scanOneBlock(const int* input, int* output, int size);
__global__ void addOffsets(int* output, int* offsets, int size);
__global__ void scanBlock(const int* input, int* output, int* blockSums, int size);

// Histogramme avec shared memory

void histogramme();
__global__ void histogrammeShared(const unsigned char* input, int* histogram, int size, int numBins);

// Stencil / Convolution avancée


// Multi-kernel pipeline


// Atomic operations et warp-synchronous programming


// Simulation simple avec forces / interactions