// ============================================================================
//  FILENAME   : main.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-12
//  UPDATED    : 2025-09-12
//  DESCRIPTION: Step 2 Cuda - Vector Addition & Memory Transfers
// ============================================================================

#include <iostream>
using namespace std;

#include <cuda_runtime.h>

__global__ void addVector(const float *A, const float *B, float *C, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
		C[idx] = A[idx] + B[idx]; 
}

int main() {
	int N = 1 << 5; // one million elements
	size_t size = N * sizeof(float);

	// CPU Memory, h for host
	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);

	 // Initialize vectors
	for (int i = 0; i < N; ++i) {
		h_A[i] = i * 2.0f;
		h_B[i] = i * 2.0f;
	}

	// GPU Memory, d for device
	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);

	// Copy from CPU to GPU
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	// Kernel lunch
	int threadPerBlock = 256;
	int blocksPerGrid = (N + threadPerBlock - 1) / threadPerBlock; // In the end n block + n2 threads on one block

	addVector<<<blocksPerGrid, threadPerBlock>>>(d_A, d_B, d_C, N);

	cudaDeviceSynchronize();

	// Copy memory from GPU to CPU
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	// Check result
	cout << "h_C[0] = " << h_C[0] << ", h_c[N - 1] = " << h_C[N - 1] << endl;

	// Free memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	return 0;
}