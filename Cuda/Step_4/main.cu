// ============================================================================
//  FILENAME   : main.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-15
//  UPDATED    : 2025-09-12
//  DESCRIPTION: Step 4 Cuda - Matrix operation - shared memory
// ============================================================================

#include <iostream>
using namespace std;

#include <cuda_runtime.h>

#define N 16
#define BLOCK_SIZE 4
#define TILE_SIZE 16

__global__ void vectorAddShared(const float *A, const float *B, float *C, int size) {
	__shared__ float sa[BLOCK_SIZE];
	__shared__ float sb[BLOCK_SIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < N) {
		// Load the shared memory
		sa[threadIdx.x] = A[tid];
		sb[threadIdx.x] = B[tid];

		// Synchronize the threads of the block
		__syncthreads();

		// COmpute and write on global memory
		C[tid] = sa[threadIdx.x] + sb[threadIdx.x];
	}
}

__global__ void matrixCrossShared(const float *A,const float *B, float *C, int size) {
	__shared__ float sa[TILE_SIZE][TILE_SIZE];
	__shared__ float sb[TILE_SIZE][TILE_SIZE];

	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	float sum = 0.0f;

	for (int t = 0; t < (size + TILE_SIZE - 1) / N; t++) {
		// Each thread stores only one element in the shared memory
		if (row < size && t * TILE_SIZE + threadIdx.x < N)
			sa[threadIdx.y][threadIdx.x] = A[row * size + t * TILE_SIZE + threadIdx.x];
		else
			sa[threadIdx.y][threadIdx.x] = 0.0f;

		if (col < size && t * TILE_SIZE + threadIdx.y < N)
			sb[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
		else
			sb[threadIdx.y][threadIdx.x] = 0.0f;

		// Waiting for all thread to compute
		__syncthreads();

		for(int k = 0; k < N; k++)
			sum += sa[threadIdx.y][k] * sb[k][threadIdx.x];
		__syncthreads();
	}

	if (col < size && row < size)
		C[row * size + col] = sum;
}

int main() {
	//////////////
	///   1D
	//////////////

	int size = N *  sizeof(float);
	
	float h_A[N], h_B[N], h_C[N];
	for(int i = 0; i < N; i++){
		h_A[i] = 2 * i;
		h_B[i] = 2 *i + 1;
	}
	cout << "A = ";
	for(int i = 0; i < N; i++) cout << h_A[i] << " ";
	cout << endl;

	cout << "B = ";
	for(int i = 0; i < N; i++) cout << h_B[i] << " ";
	cout << endl;

	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_C, size);

	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	vectorAddShared<<<(BLOCK_SIZE + N - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
	cudaDeviceSynchronize();
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	cout << "C = ";
	for(int i = 0; i < N; i++) cout << h_C[i] << " ";
	cout << endl;

	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

	//////////////
	///   2D
	//////////////

	size_t size2 = N * N * sizeof(float);

	// Allocation host
	float h_A2[N * N], h_B2[N * N], h_C2[N * N];
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++) {
			if (j == i)
				h_A2[i * N + j] = 1.0f;
			else
				h_A2[i * N + j] = 0.0f;
			h_B2[i * N + j] = 2;
		}
	}

	cout << "A = " << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			cout << h_A2[i * N + j] << " ";
		}
		cout << endl;
	}
	cout << endl;


	cout << "B = " << endl;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			cout << h_B2[i * N + j] << " ";
		}
		cout << endl;
	}
	cout << endl;


	// Allocation device
	float *d_A2, *d_B2, *d_C2;
	cudaMalloc(&d_A2, size2);
	cudaMalloc(&d_B2, size2);
	cudaMalloc(&d_C2, size2);

	// Transfert memory
	cudaMemcpy(d_A2, h_A2, size2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B2, h_B2, size2, cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);  
	dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

	matrixCrossShared<<<numBlocks, threadsPerBlock>>>(d_A2, d_B2, d_C2, N);
	cudaDeviceSynchronize();
	cudaMemcpy(h_C2, d_C2, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cout << "C =" << endl;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			cout << h_C2[i * N + j] << " ";
		cout << endl;
	}

	cudaFree(d_A2); cudaFree(d_B2); cudaFree(d_C2);

	return 0;
}