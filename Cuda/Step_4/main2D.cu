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

// #include <bits/stdc++.h>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define FILTER_RADIUS 2
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)
#define N 12  // taille de la matrice

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

int main() {
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
		
		// matrixCrossShared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
		
		// // Vérifier les erreurs CUDA
		// cudaError_t err = cudaGetLastError();
		// if (err != cudaSuccess) {
		// 	cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
		// }
		
		// cudaDeviceSynchronize();
		// cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
		
		// cout << "A × B = " << endl;
		// for (int i = 0; i < N; i++) {
		// 	for (int j = 0; j < N; j++)
		// 		cout << setw(5) << h_C[i * N + j];
		// 	cout << endl;
		// }
		// cout << endl;

		// matrixTransposeShared<<<numBlocks, threadsPerBlock>>>(d_A, d_C, N);
		// cudaDeviceSynchronize();
		// cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

		// cout << "A^T = " << endl;
		// for (int i = 0; i < N; i++) {
		// 	for (int j = 0; j < N; j++)
		// 		cout << setw(4) << h_C[i * N + j];
		// 	cout << endl;
		// }

		// matrixVectorMult<<< (N + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE >>>(d_A, d_x, d_y, N);
		// cudaDeviceSynchronize();
		// cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
		// cout << "A * x = " << endl;
		// for (int i = 0; i < N; i++) { cout << setw(4) << h_y[i] << endl; }

		// matrixVectorMultShared<<< (N + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE >>>(d_A, d_x, d_y, N);
		// cudaDeviceSynchronize();
		// cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
		// cout << "A * x (shared) = " << endl;
		// for (int i = 0; i < N; i++) { cout << setw(4) << h_y[i] << endl; }

		elementWise<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
		cudaDeviceSynchronize();
		cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
		cout << "C = sin(A) + cos(B) = " << endl;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++)
				cout << setw(8) << setprecision(3) << h_C[i * N + j];
			cout << endl;
		}
		cout << endl;	

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
	return 0;
}