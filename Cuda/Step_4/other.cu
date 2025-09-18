// ============================================================================
//  FILENAME   : main.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-16
//  UPDATED    : 2025-09-12
//  DESCRIPTION: Step 4 Cuda - Other exercices
// ============================================================================

#include <iostream>
using namespace std;

#include <cuda_runtime.h>


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


int main() {
    const int N = 1 << 10;  // 1024 éléments
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // Allocation host
    float *h_A = new float[N];
    float h_result_atomic = 0.0f;
    float h_result_shared = 0.0f;

    // Initialisation
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;  // somme attendue = N
    }

    // Allocation device
    float *d_A, *d_result, *d_partial;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMalloc((void**)&d_partial, numBlocks * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    // Kernel 1 : somme avec atomicAdd
    sumElement<<<numBlocks, blockSize>>>(d_A, N, d_result);
    cudaMemcpy(&h_result_atomic, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Kernel 2 : somme avec mémoire partagée
    sumElementShared<<<numBlocks, blockSize>>>(d_A, d_partial, N);

    // Réduire sur CPU les résultats partiels
    float *h_partial = new float[numBlocks];
    cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numBlocks; i++) {
        h_result_shared += h_partial[i];
    }

    // Affichage
    std::cout << "Somme attendue : " << N << std::endl;
    std::cout << "Somme (atomicAdd) : " << h_result_atomic << std::endl;
    std::cout << "Somme (shared memory) : " << h_result_shared << std::endl;

    // Nettoyage
    delete[] h_A;
    delete[] h_partial;
    cudaFree(d_A);
    cudaFree(d_result);
    cudaFree(d_partial);

    return 0;
}