// ============================================================================
//  FILENAME   : multikernel.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-09
//  UPDATED    : 2025-10-09
//  DESCRIPTION: Step 5 Cuda - Multikernel pipeline
// ============================================================================

#include "function.cuh"
#include "functionCPU.hpp"

__global__ void generateIntensity(float *input, int width, int height) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < height && col < width) {
		input[row * width + col] = cosf(row) * sinf(col);
	}
}

void multiKernelPipeline() {
	const int width = 1024;
	const int height = 1024;
	const int size = width * height;

	float *d_input, *d_output;
	cudaMalloc(&d_input, size * sizeof(float));
	cudaMalloc(&d_output, size * sizeof(float));

	// ---------------- Kernel generateIntensity ----------------
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	generateIntensity<<<grid, block>>>(d_input, width, height);

	// ---------------- Kernel Sobel ----------------
	float h_sobel[9] = {-1, 0, 1,
						-2, 0, 2,
						-1, 0, 1};
	float* d_sobel;
	const int sobelSize = 3;
	cudaMalloc(&d_sobel, sobelSize * sobelSize * sizeof(float));
	cudaMemcpy(d_sobel, h_sobel, sobelSize * sobelSize * sizeof(float), cudaMemcpyHostToDevice);

	dim3 block2(TILE_SIZE, TILE_SIZE);
	dim3 grid2((width + TILE_SIZE - 1)/TILE_SIZE, (height + TILE_SIZE - 1)/TILE_SIZE);
	size_t sharedMemSize = (TILE_SIZE + sobelSize - 1) * (TILE_SIZE + sobelSize - 1) * sizeof(float);

	convolutionGeneric<<<grid2, block2, sharedMemSize>>>(d_input, d_output, width, height, d_sobel, sobelSize);

	// ---------------- Threshold ----------------
	float threshold = 0.5f;
	threshold_kernel<<<grid2, block2>>>(d_output, d_output, width, height, threshold);

	// ---------------- Réduction multi-étapes ----------------
	int blockSize = 256;
	int numBlocks = (size + blockSize - 1) / blockSize;

	float* d_intermediate;
	cudaMalloc(&d_intermediate, numBlocks * sizeof(float));

	// Première étape
	reduceShared<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_output, d_intermediate, size, Add());

	// Étapes suivantes si besoin
	int remaining = numBlocks;
	while (remaining > 1) {
		int nextBlocks = (remaining + blockSize - 1) / blockSize;
		reduceShared<<<nextBlocks, blockSize, blockSize * sizeof(float)>>>(d_intermediate, d_intermediate, remaining, Add());
		remaining = nextBlocks;
	}

	cudaDeviceSynchronize();
	// Copier le résultat final
	float h_result;
	cudaMemcpy(&h_result, d_intermediate, sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "\033[33mRéduction finale : \033[0m" << h_result << std::endl;

	// ---------------- Libération mémoire ----------------
	cudaFree(d_input); cudaFree(d_output); cudaFree(d_sobel); cudaFree(d_intermediate);
}
