// ============================================================================
//  FILENAME   : main.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-03
//  UPDATED    : 2025-09-12
//  DESCRIPTION: Step 4 Cuda - Other exercices
// ============================================================================

#include <iostream>
using namespace std;

#include <bits/stdc++.h>
#include <cuda_runtime.h>

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

	float *h_A = new float[N];
	float *h_B = new float[N];
	float *h_partial = new float[numBlocks];
	float h_result = 0.0f;

	for(int i = 0; i < N; i++) {
		h_A[i] = 1.0f;
		h_B[i] = (i % 2 == 0) ? 1.0f : 0.0f;
	}

	float *d_A, *d_B, *d_partial;
	cudaMalloc((void **)&d_A, N * sizeof(float));
	cudaMalloc((void **)&d_B, N * sizeof(float));
	cudaMalloc((void **)&d_partial, numBlocks * sizeof(float));

	cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

	dotProduct<<<numBlocks, blocksize, blocksize * sizeof(float)>>>(d_A, d_B, d_partial, N);
	cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < numBlocks; i++) {
		h_result += h_partial[i];
	}

	cout << "A = "; for(int i = 0; i < N; i++) { cout << h_A[i] <<  " "; } cout << endl;
	cout << "B = "; for(int i = 0; i < N; i++) { cout << h_B[i] <<  " "; } cout << endl;
	cout << "Dot product = " << h_result << endl;

	delete[] h_A;
	delete[] h_B;
	delete[] h_partial;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_partial);

	return 0;
}