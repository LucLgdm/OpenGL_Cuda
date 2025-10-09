// ============================================================================
//  FILENAME   : reduction.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-09
//  UPDATED    : 2025-10-09
//  DESCRIPTION: Step 5 Cuda - Advanced multi-step reduction
// ============================================================================

#include "function.cuh"

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

void advancedReduction() {
	const int size = 1 << 20; // 1M elements
	const int blockSize = 256;
	const int numBlocks = (size + blockSize - 1) / blockSize;

	double* h_input = new double[size];
	for (int i = 0; i < size; ++i) {
		h_input[i] = static_cast<double>(rand()) / RAND_MAX;
	}

	double *d_input, *d_output;
	cudaMalloc(&d_input, size * sizeof(double));
	cudaMalloc(&d_output, numBlocks * sizeof(double));

	cudaMemcpy(d_input, h_input, size * sizeof(double), cudaMemcpyHostToDevice);

	reduceShared<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_input, d_output, size, Add());
	int remaining = numBlocks;
	while (remaining > 1) {
		int nextBlocks = (remaining + blockSize - 1) / blockSize;
		reduceShared<<<nextBlocks, blockSize, blockSize * sizeof(double)>>>(
				d_output, d_output, remaining, Add());
		remaining = nextBlocks;
	}
	
	double h_result;
	cudaMemcpy(&h_result, d_output, sizeof(double), cudaMemcpyDeviceToHost);

	double cpu_result = 0.0f;
	for (int i = 0; i < size; ++i) {
		cpu_result += h_input[i];
	}

	cout << "\033[32mGPU Sum: \033[0m" << h_result << endl;
	cout << "\033[32mCPU Sum: \033[0m" << cpu_result << endl;

	cudaFree(d_input); cudaFree(d_output);
	delete[] h_input;
}