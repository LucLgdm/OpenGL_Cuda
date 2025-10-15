// ============================================================================
//  FILEnAME   : atomic_Warp.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-15
//  UPDATED    : 2025-10-15
//  DESCRIPTIOn: Step 5 Cuda - atomic operation & warp-synchronous programming
// ============================================================================

#include "function.cuh"

__global__ void atomic_way(const float* data, float* result, int size) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < size) atomicMax((int*)result, __float_as_int(data[idx]));
}

__global__ void max_block(const float *input, float *maxBlock, int size) {
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	int i = blockDim.x * blockIdx.x + tid;

	sdata[tid] = (i < size) ? input[i] : -1e9f;
	__syncthreads();

	// Reduction
	for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if (tid < offset)
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
		__syncthreads();
	}

	if (tid == 0)
		maxBlock[blockIdx.x] = sdata[0];
}

// On echange directement des registres entre threads d'un meme wrap

__inline__ __device__ float warpReduceMax(float val) {
	for(int offset = 16; offset > 0; offset >>= 1)
		val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
	return val;
}

__global__ void max_warp(const float* data, float* blockMax, int size) {
	float val = -1e9f;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) val = data[i];

	// réduction intra-warp
	val = warpReduceMax(val);

	// chaque warp écrit son max dans shared memory
	__shared__ float warpMax[32]; // max 1024 threads = 32 warps
	if ((threadIdx.x & 31) == 0) warpMax[threadIdx.x >> 5] = val;
	__syncthreads();

	// le premier warp réduit les warpMax
	if (threadIdx.x < 32) {
		float local = (threadIdx.x < (blockDim.x + 31) / 32) ? warpMax[threadIdx.x] : -1e9f;
		local = warpReduceMax(local);
		if (threadIdx.x == 0) blockMax[blockIdx.x] = local;
	}
}

float maxCPU(const std::vector<float>& v) {
	return *std::max_element(v.begin(), v.end());
}


void maxValue() {
	int n = 1024 * 16;
	int blockSize = 256;
	int blockNum = (n + blockSize - 1) / blockSize;
	std::vector<float> h_data(n);
	for (int i = 0; i < n; ++i)
		h_data[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
	
	float maxCPUValue = maxCPU(h_data);
	std::cout << "Max (CPU) = " << setprecision(6) << maxCPUValue << "\n";

	float *d_data, *d_result, *d_blockMax;
	cudaMalloc(&d_data, n * sizeof(float));
	cudaMemcpy(d_data, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_result, sizeof(float));
	cudaMalloc(&d_blockMax, (n / blockSize) * sizeof(float));


	dim3 threads(blockSize);
	dim3 blocks(blockNum);

	//---------------------------------------------------------
	// Atomic
	//---------------------------------------------------------
	float zero = -1e9f;
	cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
	atomic_way<<<blocks, threads>>>(d_data, d_result, n);
	cudaDeviceSynchronize();
	float maxAtomic;
	cudaMemcpy(&maxAtomic, d_result, sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "Max (atomic) = " << setprecision(6) << maxAtomic << "\n";

	//---------------------------------------------------------
	// Shared memory reduction
	//---------------------------------------------------------
	max_block<<<blocks, threads, threads.x * sizeof(float)>>>(d_data, d_blockMax, n);
	cudaDeviceSynchronize();
	float maxBlock;
	cudaMemcpy(&maxBlock, d_result, sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "Max (block reduction) = " << setprecision(6) << maxBlock << "\n";

	//---------------------------------------------------------
	// Warp-synchronous reduction
	//---------------------------------------------------------
	max_warp<<<blocks, threads>>>(d_data, d_blockMax, n);
	cudaDeviceSynchronize();
	float maxWarp;
	cudaMemcpy(&maxWarp, d_result, sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "Max (warp reduction) = " << setprecision(6) << maxWarp << "\n";

	//---------------------------------------------------------
	// nettoyage
	//---------------------------------------------------------
	cudaFree(d_data);
	cudaFree(d_result);
	cudaFree(d_blockMax);
}