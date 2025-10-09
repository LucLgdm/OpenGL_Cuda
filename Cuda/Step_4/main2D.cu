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

#define TILE_SIZE 16
#define FILTER_RADIUS 2
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)
#define N 12  // taille de la matrice
#define N2 1024



__global__ void matrixCrossShared(const float *A, const float *B, float *C, int size) {
	__shared__ float sa[TILE_SIZE][TILE_SIZE];
	__shared__ float sb[TILE_SIZE][TILE_SIZE];
	
	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;
	float sum = 0.0f;
	
	// Nombre de tuiles nécessaires
	int numTiles = (size + TILE_SIZE - 1) / TILE_SIZE;
	
	for (int t = 0; t < numTiles; t++) {
		// Charger la tuile de A
		if (row < size && t * TILE_SIZE + threadIdx.x < size)
			sa[threadIdx.y][threadIdx.x] = A[row * size + t * TILE_SIZE + threadIdx.x];
		else
			sa[threadIdx.y][threadIdx.x] = 0.0f;
		
		// Charger la tuile de B
		if (col < size && t * TILE_SIZE + threadIdx.y < size)
			sb[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * size + col];
		else
			sb[threadIdx.y][threadIdx.x] = 0.0f;
		
		__syncthreads();
		
		// Calcul du produit partiel
		for(int k = 0; k < TILE_SIZE; k++)
			sum += sa[threadIdx.y][k] * sb[k][threadIdx.x];
		
		__syncthreads();
	}
	
	// Écrire le résultat
	if (row < size && col < size)
		C[row * size + col] = sum;
}

__global__ void matrixTransposeShared(const float *A, float *B, int size) {
	__shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 pour éviter le bank conflict

	int y = blockIdx.y * TILE_SIZE + threadIdx.y; // ligne
	int x = blockIdx.x * TILE_SIZE + threadIdx.x; // colonne

	if (y < size && x < size) {
		tile[threadIdx.y][threadIdx.x] = A[y * size + x];
	}else{
		tile[threadIdx.y][threadIdx.x] = 0.0f;
	}
	__syncthreads();

	int newX = blockIdx.y * TILE_SIZE + threadIdx.x;
	int newY = blockIdx.x * TILE_SIZE + threadIdx.y;

	if (newY < size && newX < size) {
		B[newY * size + newX] = tile[threadIdx.x][threadIdx.y];
	}
}

// A * x = y
__global__ void matrixVectorMult(const float *A, const float *x, float *y, int size) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < size) {
		float sum = 0.0f;
		for (int j = 0; j < size; j++) {
			sum += A[row * size + j] * x[j];
		}
		y[row] = sum;
	}
}

__global__ void matrixVectorMultShared(const float *A, const float *x, float *y, int size) {
	__shared__ float xshared[TILE_SIZE];

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;

	if (row >= size) return;
	for(int t = 0; t < (size + TILE_SIZE - 1) / TILE_SIZE; t++) {
		if (t * TILE_SIZE + threadIdx.x < size) {
			xshared[threadIdx.x] = x[t * TILE_SIZE + threadIdx.x];
		}else{
			xshared[threadIdx.x] = 0.0f;
		}
		__syncthreads();

		for(int j = 0; j < TILE_SIZE; j++) {
			if (t * TILE_SIZE + j < size) {
				sum += A[row * size + t * TILE_SIZE + j] * xshared[j];
			}
		}
	}
	y[row] = sum;
}

__global__ void elementWise(const float *A, const float *B, float *C, int size) {
	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;
	if (row < size && col < size)
		C[row * size + col] = cosf(A[row * size + col]) + sinf(B[row * size + col]);
}

__global__ void elementWiseShared(const float *A, const float *B, float *C, int size) {
	__shared__ float sa[TILE_SIZE][TILE_SIZE];
	__shared__ float sb[TILE_SIZE][TILE_SIZE];

	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	if (row < size && col < size) {
		sa[threadIdx.y][threadIdx.x] = A[row * size + col];
		sb[threadIdx.y][threadIdx.x] = B[row * size + col];
	}
	__syncthreads();

	if (row < size && col < size) {
		C[row * size + col] = cosf(sa[threadIdx.y][threadIdx.x]) + sinf(sb[threadIdx.y][threadIdx.x]);
	}
}

__constant__ float d_filter[FILTER_SIZE * FILTER_SIZE] = {
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f
};

__global__ void convolution2D(const float *input, float *output, int width, int height) {
	__shared__ float tile[TILE_SIZE + 2 * FILTER_RADIUS][TILE_SIZE + 2 * FILTER_RADIUS];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y * TILE_SIZE + ty;
	int col = blockIdx.x * TILE_SIZE + tx;

	// Coordonnees globales dans l'image avec le decalage
	int haloX = col - FILTER_RADIUS;
	int haloY = row - FILTER_RADIUS;

	// Charger les données dans la mémoire partagée
	if (haloX >= 0 && haloX < width && haloY >= 0 && haloY < height) {
		tile[ty][tx] = input[haloY * width + haloX];
	}else{
		tile[ty][tx] = 0.0f;
	}
	__syncthreads();

	// Appliquer le filtre
	if (ty >= FILTER_RADIUS && ty < TILE_SIZE + FILTER_RADIUS &&
		tx >= FILTER_RADIUS && tx < TILE_SIZE + FILTER_RADIUS &&
		row < height && col < width) {
			float sum = 0.0f;
			for(int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; fy++) {
				for(int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; fx++) {
					sum += tile[ty + fy][tx + fx] * d_filter[(fy + FILTER_RADIUS) * FILTER_SIZE + (fx + FILTER_RADIUS)];
				}
			}
			output[row * width + col] = sum;
	}
}

// Matrix Multiplication Benchmark
	// 1. Naive, slow baseline
__global__ void matrixMultiplication(const float *A, const float *B, float *C, int size) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < size && col < size) {
		float sum = 0.0f;
		for (int k = 0; k < size; k++) {
			sum += A[row * size + k] * B[k * size + col];
		}
		C[row * size + col] = sum;
	}
}
	// 2. Shared Memory and tiled, optimized : reuse data loaded in shared memory
__global__ void matrixMultShared(const float *A, const float *B, float *C, int size) {
	__shared__ float sa[TILE_SIZE][TILE_SIZE];
	__shared__ float sb[TILE_SIZE][TILE_SIZE];

	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	float sum = 0.0f;
	int numTiles = (size + TILE_SIZE - 1) / TILE_SIZE;
	for(int t = 0; t < numTiles; t++) {
		if (row < size && (t * TILE_SIZE + threadIdx.x) < size)
			sa[threadIdx.y][threadIdx.x] = A[row * size + t * TILE_SIZE + threadIdx.x];
		else
			sa[threadIdx.y][threadIdx.x] = 0.0f;

		if (col < size && (t * TILE_SIZE + threadIdx.y) < size)
			sb[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * size + col];
		else
			sb[threadIdx.y][threadIdx.x] = 0.0f;

		__syncthreads();

		for(int k = 0; k < TILE_SIZE; k++)
			sum += sa[threadIdx.y][k] * sb[k][threadIdx.x];
		__syncthreads();
	}
	if (row < size && col < size)
		C[row * size + col] = sum;
}
	// 3. shared memory and const __restrict__, optimized : use const memory for read-only data
__global__ void matrixMultSharedRestrict(const float *__restrict__ A, const float *__restrict__ B, float *C, int size) {
	__shared__ float sa[TILE_SIZE][TILE_SIZE];
	__shared__ float sb[TILE_SIZE][TILE_SIZE];

	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	float sum = 0.0f;
	int numTiles = (size + TILE_SIZE - 1) / TILE_SIZE;
	for(int t = 0; t < numTiles; t++) {
		if (row < size && (t * TILE_SIZE + threadIdx.x) < size)
			sa[threadIdx.y][threadIdx.x] = __ldg(&A[row * size + t * TILE_SIZE + threadIdx.x]); // __ldg for read-only cache
		else
			sa[threadIdx.y][threadIdx.x] = 0.0f;

		if (col < size && (t * TILE_SIZE + threadIdx.y) < size)
			sb[threadIdx.y][threadIdx.x] = __ldg(&B[(t * TILE_SIZE + threadIdx.y) * size + col]);
		else
			sb[threadIdx.y][threadIdx.x] = 0.0f;

		__syncthreads();

		for(int k = 0; k < TILE_SIZE; k++)
			sum += sa[threadIdx.y][k] * sb[k][threadIdx.x];
		__syncthreads();
	}
	if (row < size && col < size)
		C[row * size + col] = sum;
}

// Generic reduction kernel using shared memory
	// Op should be a functor defining the operation (e.g., addition, multiplication)
struct Add {
    __device__ float operator()(float a, float b) const { return a + b; }
	__device__ float neutral() const { return 0.0f; }
};

struct Max {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
	__device__ float neutral() const { return -FLT_MAX; }
};

struct Mul {
    __device__ float operator()(float a, float b) const { return a * b; }
	__device__ float neutral() const { return 1.0f; }
};

template<typename T, typename Op>
__global__ void reduceShared(const T* in, T* out, int n, Op op) {
	extern __shared__ T sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = (i < n) ? in[i] : op.neutral(); // Initialize with input or neutral element
	__syncthreads();

	// Perform reduction in shared memory
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s)
			sdata[tid] = op(sdata[tid], sdata[tid + s]);
		__syncthreads();
	}
	if (tid == 0) out[blockIdx.x] = sdata[0];
}


void fillMatrix(float* mat, int size) {
    for(int i = 0; i < size * size; i++)
        mat[i] = static_cast<float>(rand())/RAND_MAX;
}

void matrixMultiplicationCPU(const float* A, const float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}


int main() {
	cout << "------------------ Basic 2D operations ------------------" << endl;
	{
		size_t size = N * N * sizeof(float);
		
		// Allocation host
		float h_A[N * N], h_B[N * N], h_C[N * N];
		float h_x[N], h_y[N];
		float *d_A, *d_B, *d_C, *d_x, *d_y;
		
		// Initialisation
		if (true){
			for(int i = 0; i < N; i++){
				for(int j = 0; j < N; j++) {
					h_A[i * N + j] = (i == j) ? 1.0f : j - i;
					h_B[i * N + j] = i + j;
				}
				h_x[i] = (i == 0) ? -1.0f : .0f;
			}
			
			cout << "A = " << endl;
			for(int i = 0; i < N; i++) {
				for(int j = 0; j < N; j++)
					cout << setw(4) << h_A[i * N + j];
				cout << endl;
			}
			cout << endl;
			
			cout << "B = " << endl;
			for(int i = 0; i < N; i++) {
				for(int j = 0; j < N; j++)
					cout << setw(4) << h_B[i * N + j];
				cout << endl;
			}
			cout << endl;

			cout << "x = " << endl;
			for(int i = 0; i < N; i++) { cout << setw(4) << h_x[i] << endl; }
			cout << endl;
			
			// Allocation device
			cudaMalloc(&d_A, size);
			cudaMalloc(&d_B, size);
			cudaMalloc(&d_C, size);
			cudaMalloc(&d_x, N * sizeof(float));
			cudaMalloc(&d_y, N * sizeof(float));
			
			// Transfert mémoire
			cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
		}	
		// Configuration du kernel
		dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
		dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
		
		cout << "Matrix multiplication" << endl;
		matrixCrossShared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
		
		// Vérifier les erreurs CUDA
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
		}
		
		cudaDeviceSynchronize();
		cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
		
		cout << "A × B = " << endl;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++)
				cout << setw(5) << h_C[i * N + j];
			cout << endl;
		}
		cout << endl;

		cout << endl << "Matrix transpose" << endl;
		matrixTransposeShared<<<numBlocks, threadsPerBlock>>>(d_A, d_C, N);
		cudaDeviceSynchronize();
		cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

		cout << "A^T = " << endl;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++)
				cout << setw(4) << h_C[i * N + j];
			cout << endl;
		}

		cout << "" << endl << "Matrix-Vector multiplication" << endl;
		
		cout << "	Non-shared version" << endl;
		matrixVectorMult<<< (N + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE >>>(d_A, d_x, d_y, N);
		cudaDeviceSynchronize();
		cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "A * x = " << endl;
		for (int i = 0; i < N; i++) { cout << setw(4) << h_y[i] << endl; }
		
		cout << "	Shared version" << endl;
		matrixVectorMultShared<<< (N + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE >>>(d_A, d_x, d_y, N);
		cudaDeviceSynchronize();
		cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "A * x (shared) = " << endl;
		for (int i = 0; i < N; i++) { cout << setw(4) << h_y[i] << endl; }

		cout << endl << "Element-wise operations" << endl;
		cout << "	Non-shared version" << endl;
		elementWise<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
		cudaDeviceSynchronize();
		cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
		cout << "C = sin(A) + cos(B) = " << endl;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++)
				cout << setw(8) << setprecision(3) << h_C[i * N + j];
			cout << endl;
		}
		
		cout << endl << "	Shared version" << endl;
		elementWiseShared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
		cudaDeviceSynchronize();
		cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
		cout << "C (shared) = sin(A) + cos(B) = " << endl;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++)
				cout << setw(8) << setprecision(3) << h_C[i * N + j];
			cout << endl;
		}
		cout << endl;
		
		cudaFree(d_A); 
		cudaFree(d_B); 
		cudaFree(d_C);
		cudaFree(d_x);
		cudaFree(d_y);
	}
	cout << endl <<  "------------------ Convolution 2D ------------------" << endl;
	{
		const int width = 7;
		const int height = 7;

		float h_input[width * height] = {
			1, 2, 3, 4, 5, 6, 7,
			7, 6, 5, 4, 3, 2, 1,
			1, 2, 3, 4, 5, 6, 7,
			7, 6, 5, 4, 3, 2, 1,
			1, 1, 1, 1, 1, 1, 1,
			2, 2, 2, 2, 2, 2, 2,
			3, 3, 3, 3, 3, 3, 3
		};
		float h_output[width * height] = {0};

		float *d_input, *d_output;
		cudaMalloc(&d_input, width * height * sizeof(float));
		cudaMalloc(&d_output, width * height * sizeof(float));

		cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemset(d_output, 0, width * height * sizeof(float));

		dim3 threads(TILE_SIZE + 2*FILTER_RADIUS, TILE_SIZE + 2*FILTER_RADIUS);
		dim3 blocks((width + TILE_SIZE - 1)/TILE_SIZE, (height + TILE_SIZE - 1)/TILE_SIZE);

		convolution2D<<<blocks, threads>>>(d_input, d_output, width, height);
		cudaDeviceSynchronize();

		cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

		std::cout << "Input image:\n";
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++)
				std::cout << h_input[y*width + x] << "\t";
			std::cout << "\n";
		}

		std::cout << "\nOutput after convolution:\n";
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++)
				std::cout << h_output[y*width + x] << "\t";
			std::cout << "\n";
		}

		cudaFree(d_input);
		cudaFree(d_output);
	}
	cout << endl <<  "------------------ Matrix Multiplication Benchmark ------------------" << endl;
	{
		int size = N2;
		size_t bytes = size*size*sizeof(float);

		// Host matrices
		float *h_A = new float[size*size];
		float *h_B = new float[size*size];
		float *h_C = new float[size*size];
		float *h_C_cpu = new float[size*size];

		fillMatrix(h_A, size);
		fillMatrix(h_B, size);

		// Device matrices
		float *d_A, *d_B, *d_C;
		cudaMalloc(&d_A, bytes);
		cudaMalloc(&d_B, bytes);
		cudaMalloc(&d_C, bytes);

		cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

		dim3 threads(TILE_SIZE, TILE_SIZE);
		dim3 blocks((size + TILE_SIZE -1)/TILE_SIZE, (size + TILE_SIZE -1)/TILE_SIZE);

		// CPU timing
		auto start_cpu = chrono::high_resolution_clock::now();
		matrixMultiplicationCPU(h_A, h_B, h_C_cpu, N2);
		auto end_cpu = chrono::high_resolution_clock::now();
		chrono::duration<double, milli> cpu_time = end_cpu - start_cpu;

		cout << "For 1024 * 1024 elements:" << endl;
		cout << "Naive CPU: " << cpu_time.count() << " ms" << std::endl;

		// GPU timing
		cudaEvent_t start, stop;
		float ms;

		// -------- Naive --------
		cudaEventCreate(&start); cudaEventCreate(&stop);
		cudaEventRecord(start);
		matrixMultiplication<<<blocks, threads>>>(d_A, d_B, d_C, size);
		cudaEventRecord(stop); cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		std::cout << "Naive GPU: " << ms << " ms\n";

		// -------- Shared --------
		cudaEventRecord(start);
		matrixMultShared<<<blocks, threads>>>(d_A, d_B, d_C, size);
		cudaEventRecord(stop); cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		std::cout << "Shared memory GPU: " << ms << " ms\n";

		// -------- Shared + __restrict__ --------
		cudaEventRecord(start);
		matrixMultSharedRestrict<<<blocks, threads>>>(d_A, d_B, d_C, size);
		cudaEventRecord(stop); cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		std::cout << "Shared + __restrict__ GPU: " << ms << " ms\n";

		// Cleanup
		cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
		delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_cpu;
		cudaEventDestroy(start); cudaEventDestroy(stop);
	}
	cout << endl <<  "------------------ Reduction with shared memory ------------------" << endl;
	{
		int size = 9;
		size_t bytes = size*size*sizeof(float);

		float h_in[size*size];
		fillMatrix(h_in, size);
		// for(int i = 0; i < size*size; i++) h_in[i] = (i + 1) * 1.0f;

		float *d_in, *d_out;
		cudaMalloc(&d_in, bytes);
		cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

		int blockSize = 4;
		int numBlocks = (size + blockSize - 1) / blockSize;
		cudaMalloc(&d_out, numBlocks * sizeof(float));

		reduceShared<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, size, Add());

		// Deuxième passe
		int s = numBlocks;
		while (s > 1) {
			int nb = (s + blockSize - 1) / blockSize;
			reduceShared<<<nb, blockSize, blockSize * sizeof(float)>>>(d_out, d_out, s, Add());
			s = nb;
		}

		float result;
		cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
		std::cout << "Result = " << result << std::endl;

		cudaFree(d_in); cudaFree(d_out);
	}
	cout << endl << "------------------ End ------------------" << endl;
	return 0;
}