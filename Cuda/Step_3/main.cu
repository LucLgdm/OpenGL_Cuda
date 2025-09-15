// ============================================================================
//  FILENAME   : main.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-13
//  UPDATED    : 2025-09-12
//  DESCRIPTION: Step 3 Cuda - 2D/3D Grids & Memory Hierarchy
// ============================================================================

#include <stdio.h>
#include <cuda_runtime.h>

#include <iostream>
using namespace std;

#define N 4

__global__ void matrixAdd(const float *A, const float *B, float *C, int size) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < size && col < size){
		int idx = row * size + col;
		C[idx] = A[idx] + B[idx];
		// printf("Thread (%d, %d) -> C[%d] = %f\n", row, col, idx, C[idx]);
	}
}


__global__ void matrixCross(const float *A, const float *B, float *C, int size) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < size && col < size) {
		int idx = row * size + col;
		float sum = 0.0f;
		for(int k = 0; k < size; k++) {
			sum += A[row * size + k] * B[k * size + col];
		}
		C[idx] = sum;
		// printf("Thread (%d, %d) -> C[%d] = %f\n", row, col, idx, C[idx]);
	}
}


int main() {
	size_t size = N * N * sizeof(float);

	// Allocation host
	float h_A[N * N], h_B[N * N], h_C[N * N];
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++) {
			if (j == i)
				h_A[i * N + j] = 1.0f;
			else
				h_A[i * N + j] = 0.0f;
			h_B[i * N + j] = 2;
		}
	}

	// Allocation device
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_C, size);

	// Transfert memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	// Definition of block and grid
	dim3 threadsPerBlock(2, 2);
	dim3 numBlocks((N + 1) / 2, (N + 1) / 2);

	// Lancemenmt du kernel
	// matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
	matrixCross<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
	cudaDeviceSynchronize();

	// Transfert memory
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


	cout << "A = " << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			cout << h_A[i * N + j] << " ";
		}
		cout << endl;
	}


	cout << "B = " << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			cout << h_B[i * N + j] << " ";
		}
		cout << endl;
	}

	cout << "C = " << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			cout << h_C[i * N + j] << " ";
		}
		cout << endl;
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}