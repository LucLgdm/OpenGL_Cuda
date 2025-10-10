// ============================================================================
//  FILENAME   : histogramme.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-09
//  UPDATED    : 2025-10-10
//  DESCRIPTION: Step 5 Cuda - Histogramme with shared memory
// ============================================================================

#include "function.cuh"

// Histogramme naif, utile pour petit nombre de valeurs
__global__ void histNaive(const unsigned int *input, int *histogram, int size, int BIN_COUNT) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		int histIndex = input[idx];
		if (histIndex < BIN_COUNT) {
			atomicAdd(&histogram[histIndex], 1);
		}
	}
}

// Histogramme avec shared memory, on calcule un histogramme par bloc
__global__ void histShared(const unsigned int *input, int *histogram, int size, int BIN_COUNT) {
	extern __shared__ int sHist[];
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + tid;
	
	for(int i = tid; i < BIN_COUNT; i += blockDim.x) {
		sHist[i] = 0;
	}
	__syncthreads();

	if (idx < size) {
		int bin = input[idx];
		if (bin < BIN_COUNT) {
			atomicAdd(&sHist[bin], 1);
		}
	}
	__syncthreads();

	for(int i = tid; i < BIN_COUNT; i += blockDim.x) {
		atomicAdd(&histogram[i], sHist[i]);
	}
}

__inline__ __device__ int warpId() {
	return threadIdx.x / warpSize;
}

__inline__ __device__ int laneId() {
	return threadIdx.x % warpSize;
}

// Histogramme avec shared memory et warps, on calcule un histogramme par warp :
// 32 threads tapent sur les memes cases et non 256 pour un bloc, ca réduit les conflits
// On reduit tous les histogrammes de warps dans un histogramme de bloc avant d'écrire en global
__global__ void histWarpShared(const unsigned int *input, int *histogram, int size, int BIN_COUNT) {
	// Contient un histogramme par warp
	extern __shared__ int sHist[];

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + tid;
	int numWarps = blockDim.x / warpSize;
	int wid = warpId(); // warp index
	int lid = laneId(); // indice du thread dans le warp

	// Chaque warp initialise son histogramme dans la shared memory
	int *sWarpHist = sHist + wid * BIN_COUNT;

	for(int i = lid; i < BIN_COUNT; i += warpSize) {
		sWarpHist[i] = 0;
	}
	__syncthreads();

	// Accumulation locale
	if (idx < size) {
		int bin = input[idx];
		if (bin < BIN_COUNT) {
			atomicAdd(&sWarpHist[bin], 1);
		}
	}
	__syncthreads();

	// Réduction des histogrammes de warps dans l'histogramme de bloc
	for(int bin = tid; bin < BIN_COUNT; bin += blockDim.x) {
		int sum = 0;
		// On somme sur tous les warps du bloc
		for(int w = 0; w < numWarps; w++) {
			sum += sHist[w * BIN_COUNT + bin];
		}
		sHist[bin] = sum;
	}
	__syncthreads();

	// Écriture dans l'histogramme global
	for(int bin = tid; bin < BIN_COUNT; bin += blockDim.x) {
		atomicAdd(&histogram[bin], sHist[bin]);
	}
}

void histogramme() {
	const int SIZE = 1 << 21;
	const int BLOCK_SIZE = 256;
	const int NUM_BLOCKS = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

	for(int BIN_COUNT : {16, 32, 64, 128, 256, 512, 1024}) {

		// Allocation et initialisation du tableau d’entrée
		vector<unsigned int> h_input(SIZE);
		for (int i = 0; i < SIZE; ++i)
			h_input[i] = rand() % BIN_COUNT;

		// Allocation device
		unsigned int *d_input;
		int *d_histNaive, *d_histShared, *d_histWarpShared;
		cudaMalloc(&d_input, SIZE * sizeof(unsigned int));
		cudaMalloc(&d_histNaive, BIN_COUNT * sizeof(int));
		cudaMalloc(&d_histShared, BIN_COUNT * sizeof(int));
		cudaMalloc(&d_histWarpShared, BIN_COUNT * sizeof(int));

		cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemset(d_histNaive, 0, BIN_COUNT * sizeof(int));
		cudaMemset(d_histShared, 0, BIN_COUNT * sizeof(int));
		cudaMemset(d_histWarpShared, 0, BIN_COUNT * sizeof(int));

		// -----------------------------
		//   Kernel Naïf
		// -----------------------------
		auto start = chrono::high_resolution_clock::now();
		histNaive<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_histNaive, SIZE, BIN_COUNT);
		cudaDeviceSynchronize();
		auto end = chrono::high_resolution_clock::now();
		double tNaive = chrono::duration<double, milli>(end - start).count();

		vector<int> h_histNaive(BIN_COUNT);
		cudaMemcpy(h_histNaive.data(), d_histNaive, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

		// -----------------------------
		//   Kernel Shared
		// -----------------------------
		cudaMemset(d_histShared, 0, BIN_COUNT * sizeof(int));
		size_t sharedBytes = BIN_COUNT * sizeof(int);

		start = chrono::high_resolution_clock::now();
		histShared<<<NUM_BLOCKS, BLOCK_SIZE, sharedBytes>>>(d_input, d_histShared, SIZE, BIN_COUNT);
		cudaDeviceSynchronize();
		end = chrono::high_resolution_clock::now();
		double tShared = chrono::duration<double, milli>(end - start).count();

		vector<int> h_histShared(BIN_COUNT);
		cudaMemcpy(h_histShared.data(), d_histShared, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

		// -----------------------------
		//   Kernel Warp Shared
		// -----------------------------
		cudaMemset(d_histWarpShared, 0, BIN_COUNT * sizeof(int));
		sharedBytes = (BLOCK_SIZE / 32) * BIN_COUNT * sizeof(int); // un histogramme par warp

		start = chrono::high_resolution_clock::now();
		histWarpShared<<<NUM_BLOCKS, BLOCK_SIZE, sharedBytes>>>(d_input, d_histWarpShared, SIZE, BIN_COUNT);
		cudaDeviceSynchronize();
		end = chrono::high_resolution_clock::now();
		double tWarpShared = chrono::duration<double, milli>(end - start).count();

		vector<int> h_histWarpShared(BIN_COUNT);
		cudaMemcpy(h_histWarpShared.data(), d_histWarpShared, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

		// -----------------------------
		// Vérification CPU
		// -----------------------------
		vector<int> h_ref(BIN_COUNT, 0);
		for (int i = 0; i < SIZE; ++i)
			h_ref[h_input[i]]++;

		bool ok1 = true, ok2 = true, ok3 = true;
		for (int i = 0; i < BIN_COUNT; ++i) {
			if (h_histNaive[i] != h_ref[i]) ok1 = false;
			if (h_histShared[i] != h_ref[i]) ok2 = false;
			if (h_histWarpShared[i] != h_ref[i]) ok3 = false;
		}

		// -----------------------------
		// Résultats
		// -----------------------------
		cout << "\033[36m   Formation of histogram with " << SIZE << " values and " << BIN_COUNT << " bins\033[0m" << endl;
		cout << "\033[32mNaive Kernel      : \033[0m" << tNaive << " ms"
				<< " | Correct: " << (ok1 ? "\033[32mYES\033[0m" : "\033[33mNO\033[0m") << endl;
		cout << "\033[32mShared Kernel     : \033[0m" << tShared << " ms"
				<< " | Correct: " << (ok2 ? "\033[32mYES\033[0m" : "\033[33mNO\033[0m") << endl;
		cout << "\033[32mWarp Shared Kernel: \033[0m" << tWarpShared << " ms"
				<< " | Correct: " << (ok3 ? "\033[32mYES\033[0m" : "\033[33mNO\033[0m") << endl;
		cout << "\n\033[32mHistogram    : \033[0m";
		for (int i = 0; i < BIN_COUNT; ++i)
			cout << h_histShared[i] << " ";
		cout << "\n\n";

		cudaFree(d_input);
		cudaFree(d_histNaive);
		cudaFree(d_histShared);
		cudaFree(d_histWarpShared);
	}
}