// ============================================================================
//  FILENAME   : scan.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-09
//  UPDATED    : 2025-10-09
//  DESCRIPTION: Step 5 Cuda - Scan (Prefix Sum)
// ============================================================================

#include "function.cuh"

__global__ void scanOneBlock(const int* input, int* output, int size) {
	extern __shared__ int temp[];
	int thid = threadIdx.x;
	int offset = 1;

	// Load input into shared memory
	temp[thid] = (thid < size) ? input[thid] : 0;
	__syncthreads();

	// Up-sweep (reduce) phase
	for(int d = size >> 1; d > 0; d >>= 1){
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset <<= 1; // *= 2
	}
	// Clear the last element
	if (thid == 0) {
		temp[size - 1] = 0;
	}
	__syncthreads();

	// Down-sweep phase
	for(int d = 1; d < size; d <<= 1) {
		offset >>= 1;
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	// // Exclusive way
	// if (thid < size) {
	// 	output[thid] = temp[thid];
	// }

	// Inclusive way
	if (thid < size) {
		output[thid] = temp[thid] + input[thid];
	}
}

__global__ void addOffsets(int* output, int* offsets, int size) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= size) return;
	output[gid] += offsets[blockIdx.x];
}

__global__ void scanBlock(const int* input, int* output, int* blockSums, int size) {
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + thid;
    int offset = 1;

    // Charger les éléments valides
    temp[thid] = (gid < size) ? input[gid] : 0;
    __syncthreads();

    // Up-sweep
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2*thid + 1) - 1;
            int bi = offset * (2*thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Clear last for exclusive scan
    if (thid == 0) blockSums[blockIdx.x] = temp[blockDim.x - 1]; // store sum
    temp[blockDim.x - 1] = 0;
    __syncthreads();

    // Down-sweep
    for (int d = 1; d < blockDim.x; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid +1)-1;
            int bi = offset*(2*thid +2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Écrire le résultat final (scan inclusif)
    if (gid < size)
        output[gid] = temp[thid] + input[gid];
}


void scan(){
	cout << "\033[36m	--- Scan one block ---\033[0m" << endl;
	scanLonelyBlock();
	cout << "\033[36m	--- Scan multi-block ---\033[0m" << endl;
	scanMultiBlock();
}

void scanLonelyBlock() {
	const int size = 1 << 6;
	const int blockSize = 1024;
	int* h_input = new int[size];
	for (int i = 0; i < size; ++i) {
		h_input[i] = static_cast<int>(rand() % 10);
	}
	int *d_input, *d_output;
	cudaMalloc(&d_input, size * sizeof(int));
	cudaMalloc(&d_output, size * sizeof(int));
	cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);


	int sharedMemSize = blockSize * sizeof(int);
	scanOneBlock<<<1, blockSize, sharedMemSize>>>(d_input, d_output, size);

	int* h_output = new int[size];
	cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
	cout << "\033[32mInput : \033[0m";
	for (int i = 0; i < size; ++i) { cout << setw(4) << h_input[i]; } cout << endl;
	cout << "\033[32mOutput: \033[0m";
	for (int i = 0; i < size; ++i) { cout << setw(4) << h_output[i]; } cout << endl;

	cudaFree(d_input); cudaFree(d_output);
	delete[] h_input; delete[] h_output;
}

void scanMultiBlock() {
	const int size = 1 << 20;
	const int blockSize = 1024;
	int numBlocks = (size + blockSize -1) / blockSize;

	int* h_input = new int[size];
	for (int i = 0; i < size; ++i) {
		h_input[i] = static_cast<int>(rand() % 10);
	}
	int *d_input, *d_output;
	cudaMalloc(&d_input, size * sizeof(int));
	cudaMalloc(&d_output, size * sizeof(int));
	cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

	int* d_blockSums;
	cudaMalloc(&d_blockSums, numBlocks * sizeof(int));

	// Scan local par bloc
	scanBlock<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, d_blockSums, size);

	if (numBlocks > 1) {
		// Scan du tableau des sommes de bloc
		int* d_blockOffsets;
		cudaMalloc(&d_blockOffsets, numBlocks * sizeof(int));
		scanBlock<<<1, numBlocks, numBlocks * sizeof(int)>>>(d_blockSums, d_blockOffsets, d_blockSums, numBlocks);

		// Appliquer les offsets à chaque bloc
		addOffsets<<<numBlocks, blockSize>>>(d_output, d_blockOffsets, size);
		cudaFree(d_blockOffsets);
	}

	 int* h_output = new int[size];
    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

	cout << "\033[32mInput : \033[0m";
	for (int i = 0; i < 50; ++i) { cout << setw(4) << h_input[i]; } cout << " ..." << endl;
    cout << "\033[32mOutput: \033[0m";
    for (int i = 0; i < 50; ++i) { cout << setw(4) << h_output[i]; } cout << " ..." << endl;

    cudaFree(d_input); 
    cudaFree(d_output);
    cudaFree(d_blockSums);
    delete[] h_input; 
    delete[] h_output;
}