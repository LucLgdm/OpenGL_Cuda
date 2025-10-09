// ============================================================================
//  FILENAME   : main.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-15
//  UPDATED    : 2025-09-12
//  DESCRIPTION: Step 4 Cuda - operation 1D - shared memory
// ============================================================================

#include <iostream>
using namespace std;

#include <bits/stdc++.h>
#include <cuda_runtime.h>

__global__ void vectorAddShared(const float *A, const float *B, float *C, int n) {
	__shared__ float sa[256]; // Block size
	__shared__ float sb[256]; // Block size

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n) {
		// Load the shared memory
		sa[threadIdx.x] = A[tid];
		sb[threadIdx.x] = B[tid];

		// Synchronize the threads of the block
		__syncthreads();

		// COmpute and write on global memory
		C[tid] = sa[threadIdx.x] + sb[threadIdx.x];
	}
}

__global__ void sumElement(const float *A, int n, float *result) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n)
		atomicAdd(result, A[idx]); // Avoid the data races : threads writing at the same time
}

__global__ void sumElementShared(const float *A, float *partial, int n) {
	__shared__ float cache[256];

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Load one element per thread
	float temp = (idx < n) ? A[idx] : 0.0f;
	cache[tid] = temp;

	__syncthreads();

	for(int stride = blockDim.x /2; stride > 0; stride >>= 1) {
		if (tid < stride)
			cache[tid] += cache[tid + stride];
		__syncthreads();
	}

	if (tid == 0)
		partial[blockIdx.x] = cache[0];
}

__global__ void dotProduct(const float *A, const float *B, float *partial, int n) {
    extern __shared__ float cache[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = (idx < n) ? A[idx] * B[idx] : 0.0f;
    cache[tid] = temp;

    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            cache[tid] += cache[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = cache[0];
}


int main() {
	const int N = 1 << 3;
	const int blocksize = 256;
	const int numBlocks = (N + blocksize - 1) / blocksize;

	int size = N * sizeof(float);
	
	float *h_A = new float[N];
	float *h_B = new float[N];
	float *h_C = new float[N];
	float *h_partial = new float[numBlocks];
	float h_result = 0.0f;
	float h_result_atomic = 0.0f;  // AJOUT
	float h_result_shared = 0.0f;  // AJOUT

	for(int i = 0; i < N; i++) {
		h_A[i] = 1.0f;
		h_B[i] = (i % 2 == 0) ? 1.0f : 0.0f;
	}
	cout << "A     = "; for(int i = 0; i < N; i++) { cout << setw(3) << h_A[i] <<  " "; } cout << endl;
	cout << "B     = "; for(int i = 0; i < N; i++) { cout << setw(3) << h_B[i] <<  " "; } cout << endl;

	float *d_A, *d_B, *d_C, *d_partial, *d_result;  // AJOUT d_result
	cudaMalloc((void **)&d_A, N * sizeof(float));
	cudaMalloc((void **)&d_B, N * sizeof(float));
	cudaMalloc((void **)&d_C, N * sizeof(float));
	cudaMalloc((void **)&d_partial, numBlocks * sizeof(float));
	cudaMalloc((void **)&d_result, sizeof(float));  // AJOUT
	cudaMemset(d_result, 0, sizeof(float));  // AJOUT - important pour atomicAdd

	cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

	cout << "Dot product with shared memory" << endl;
	dotProduct<<<numBlocks, blocksize, blocksize * sizeof(float)>>>(d_A, d_B, d_partial, N);
	cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < numBlocks; i++) {
		h_result += h_partial[i];
	}
	cout << "Result = " << h_result << endl << endl;

	cout << "A + B with shared memory" << endl;
	// Modifier les données
	for(int i = 0; i < N; i++){
		h_A[i] = 2 * i;
		h_B[i] = 2 * i + 1;
	}
	cout << "A     = ";
	for(int i = 0; i < N; i++) cout << setw(3) << h_A[i] << " ";
	cout << endl;

	cout << "B     = ";
	for(int i = 0; i < N; i++) cout << setw(3) << h_B[i] << " ";
	cout << endl;

	// AJOUT : recopier les nouvelles données vers le GPU
	cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

	vectorAddShared<<<numBlocks, blocksize>>>(d_A, d_B, d_C, N);  // FIX: numBlocks au lieu de (blocksize + N - 1) / blocksize
	cudaDeviceSynchronize();
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	cout << "A + B = ";
	for(int i = 0; i < N; i++) cout << setw(3) << h_C[i] << " ";
	cout << endl;

	cout << endl << "Sum of elements of :" << endl;
	cout << "A     = "; for(int i = 0; i < N; i++) { cout << setw(3) << h_A[i] <<  " "; } cout << endl;
	// Kernel 1 : somme avec atomicAdd
	sumElement<<<numBlocks, blocksize>>>(d_A, N, d_result);  // FIX: blocksize
	cudaMemcpy(&h_result_atomic, d_result, sizeof(float), cudaMemcpyDeviceToHost);

	// Kernel 2 : somme avec mémoire partagée
	sumElementShared<<<numBlocks, blocksize>>>(d_A, d_partial, N);  // FIX: blocksize

	// Réduire sur CPU les résultats partiels
	cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < numBlocks; i++) {
		h_result_shared += h_partial[i];
	}

	// Affichage
	float expected_sum = 0;
	for(int i = 0; i < N; i++) expected_sum += h_A[i];
	cout << "Somme attendue 		: " << expected_sum << endl;
	cout << "Somme (atomicAdd) 	: " << h_result_atomic << endl;
	cout << "Somme (shared memory) 	: " << h_result_shared << endl;

	delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_partial;
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_partial);
	cudaFree(d_result);
	return 0;
}