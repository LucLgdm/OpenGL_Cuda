// ============================================================================
//  FILENAME   : Stencil_Convolution.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-09
//  UPDATED    : 2025-10-09
//  DESCRIPTION: Step 5 Cuda - Stencil / Advanced Convolution
// ============================================================================

#include "function.cuh"

__global__ void convolutionGeneric(const float*input, float *output, int width, int height,
			const float *filter, int filterSize) {
	extern __shared__ float stile[];

	int radius = filterSize / 2;
	int tx = threadIdx.x; // Indice du thread dans le bloc x et y
	int ty = threadIdx.y;
	int col = blockIdx.x * blockDim.x + tx; // De l'image globale
	int row = blockIdx.y * blockDim.y + ty;

	int tileW = TILE_SIZE + 2 * radius; // Taille du tile avec les bordures
	int haloX = col - radius; // Coordonnee globale du pixel en tenant compte de la bordure
	int haloY = row - radius;

	if (haloX >= 0 && haloX < width && haloY >= 0 && haloY < height) {
		stile[ty * tileW + tx] = input[haloY * width + haloX];
	}else{
		stile[ty * tileW + tx] = 0.0f;
	}
	__syncthreads();

	// Appliquer le filtre
	if (ty >= radius && ty < TILE_SIZE + radius &&
		tx >= radius && tx < TILE_SIZE + radius &&
		col < height && row < width) {

			float sum = 0.0f;

			/*
			Filtre :           Partie de l'image sur laquelle on applique :
				1 2 3              a b c
				4 5 6   ×          d e f
				7 8 9              g h i
			Pour le pixel e, on fait : sum = 1*a + 2*b + 3*c + 4*d + 5*e + 6*f + 7*g + 8*h + 9*i
			*/
			for(int fy = -radius; fy <= radius; fy++) {
				for(int fx = -radius; fx <= radius; fx++) {
					sum += filter[(fy + radius) * filterSize + (fx + radius)] *
							stile[(ty + fy) * tileW + (tx + fx)];
				}
			}
			// Ecrire le resultat dans l'image de sortie
			output[row * width + col] = sum;
		}

}

__global__ void threshold_kernel(const float* input, float* output,
								int width, int height, float threshold) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	float val = input[y * width + x];
	output[y * width + x] = (val > threshold ? 1.0f : 0.0f);
}


void pipeline() {
	const int width = 2048;
	const int height = 2048;
	// === Événements pour mesurer multiConvolution ===
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);

	cudaEventRecord(start);
	multiConvolution(width, height);  // version séquentielle
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float timeSequential = 0.0f;
	cudaEventElapsedTime(&timeSequential, start, stop);

	cout << "	\033[33m-------------------\033[0m" << endl;
	// === Événements pour mesurer overlapping ===
	cudaEventRecord(start);
	overlapping(width, height);  // version avec streams et overlap
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float timeOverlap = 0.0f;
	cudaEventElapsedTime(&timeOverlap, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	// === Comparaison finale ===
	cout << endl << "\033[1m\033[36mComparaison des performances a taille egale ("<< width << " * " << height << "):\033[0m" << endl;
	cout << "  • \033[33mPipeline classique : \033[0m" << timeSequential << " ms" << endl;
	cout << "  • \033[33mPipeline overlapping : \033[0m" << timeOverlap << " ms" << endl;
	cout << "  • \033[33mGain ≈ \033[0mx" << fixed << setprecision(2)
	     << (timeSequential / timeOverlap) << endl;
}


void multiConvolution(const int width, const int height) {
	const int size = width * height;

	// ---------------- Image simulee ----------------
	float* h_input = new float[size];
	for (int i = 0; i < size; ++i)
		h_input[i] = static_cast<float>(rand() % 10) / 10.0f; // valeurs 0.0 à 0.9

	// ---------------- Filtres ----------------
	const int gaussianSize = 3;
	float h_gaussian[gaussianSize * gaussianSize] = {
		1/16.f, 2/16.f, 1/16.f,
		2/16.f, 4/16.f, 2/16.f,
		1/16.f, 2/16.f, 1/16.f
	};

	const int sobelSize = 3;
	float h_sobel[sobelSize * sobelSize] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1
	};

	// ---------------- Allocation GPU ----------------
	float *d_input, *d_temp, *d_output;
	float *d_gaussian, *d_sobel;
	cudaMalloc(&d_input, size * sizeof(float));
	cudaMalloc(&d_temp, size * sizeof(float));
	cudaMalloc(&d_output, size * sizeof(float));
	cudaMalloc(&d_gaussian, gaussianSize * gaussianSize * sizeof(float));
	cudaMalloc(&d_sobel, sobelSize * sobelSize * sizeof(float));

	cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gaussian, h_gaussian, gaussianSize * gaussianSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sobel, h_sobel, sobelSize * sobelSize * sizeof(float), cudaMemcpyHostToDevice);

	// ---------------- Dimension blocks / grid ----------------
	dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid((width + TILE_SIZE -1)/TILE_SIZE, (height + TILE_SIZE -1)/TILE_SIZE);

	// ---------------- Pipeline ----------------

	// Gaussian
	size_t sharedMemSize = (TILE_SIZE + gaussianSize - 1) * (TILE_SIZE + gaussianSize - 1) * sizeof(float);
	convolutionGeneric<<<grid, block, sharedMemSize>>>(d_input, d_temp, width, height, d_gaussian, gaussianSize);
	cudaDeviceSynchronize();

	// Sobel
	sharedMemSize = (TILE_SIZE + sobelSize - 1) * (TILE_SIZE + sobelSize - 1) * sizeof(float);
	convolutionGeneric<<<grid, block, sharedMemSize>>>(d_temp, d_temp, width, height, d_sobel, sobelSize);
	cudaDeviceSynchronize();

	// Threshold
	float threshold = 0.5f;
	threshold_kernel<<<grid, block>>>(d_temp, d_output, width, height, threshold);
	cudaDeviceSynchronize();

	// ---------------- Copier resultat sur hôte ----------------
	float* h_output = new float[size];
	cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

	// ---------------- Affichage console ----------------
	cout << "    \033[36m\033[3mImage avant pipeline (extrait 10x10)\033[0m" << endl;
	for (int y = 0; y < 10; ++y) {
		for (int x = 0; x < 10; ++x)
			cout << fixed << setprecision(1) << h_input[y * width + x] << " ";
		cout << endl;
	}

	cout << "    \033[36m\033[3mImage après pipeline\033[0m" << endl;
	for (int y = 0; y < 10; ++y) {
		for (int x = 0; x < 10; ++x)
			cout << fixed << setprecision(1) << h_output[y * width + x] << " ";
		cout << endl;
	}

	// ---------------- Liberation memoire ----------------
	cudaFree(d_input); cudaFree(d_temp); cudaFree(d_output);
	cudaFree(d_gaussian); cudaFree(d_sobel);
	delete[] h_input; delete[] h_output;
}

void overlapping(const int width, const int height) {
	const int size = width * height;
	const int bandHeight = 256;
	const int numBands = (height + bandHeight - 1) / bandHeight;

	// ---------------- Image simulee ----------------
	float* h_input = new float[size];
	float* h_output = new float[size];
	for (int i = 0; i < size; ++i)
		h_input[i] = static_cast<float>(rand() % 10) / 10.0f;

	// ---------------- Filtres ----------------
	const int gaussianSize = 3;
	float h_gaussian[gaussianSize * gaussianSize] = {
		1/16.f, 2/16.f, 1/16.f,
		2/16.f, 4/16.f, 2/16.f,
		1/16.f, 2/16.f, 1/16.f
	};
	const int sobelSize = 3;
	float h_sobel[sobelSize * sobelSize] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1
	};

	// ---------------- Allocation GPU ----------------
	float *d_input, *d_output, *d_temp;
	float *d_gaussian, *d_sobel;
	cudaMalloc(&d_input, size * sizeof(float));
	cudaMalloc(&d_output, size * sizeof(float));
	cudaMalloc(&d_temp, size * sizeof(float));
	cudaMalloc(&d_gaussian, gaussianSize * gaussianSize * sizeof(float));
	cudaMalloc(&d_sobel, sobelSize * sobelSize * sizeof(float));

	cudaMemcpy(d_gaussian, h_gaussian, gaussianSize * gaussianSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sobel, h_sobel, sobelSize * sobelSize * sizeof(float), cudaMemcpyHostToDevice);

	// ---------------- overlapping ----------------
	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);

	for(int b = 0; b < numBands; ++b) {
		int startRow = b * bandHeight;
		int rows = min(bandHeight, height - startRow);

		cudaStream_t s = streams[b % 2];

		// 1 Copier la bande Hote -> Device
		cudaMemcpyAsync(d_input + startRow * width,
						h_input + startRow * width,
						rows * width * sizeof(float),
						cudaMemcpyHostToDevice, s);

		dim3 block(TILE_SIZE, TILE_SIZE);
		dim3 grid((width + TILE_SIZE - 1)/TILE_SIZE, (rows + TILE_SIZE - 1)/TILE_SIZE);

		// 2 Gaussien
		size_t sharedMemSize = (TILE_SIZE + gaussianSize - 1) * (TILE_SIZE + gaussianSize - 1) * sizeof(float);
		convolutionGeneric<<<grid, block, sharedMemSize, s>>>(
			d_input + startRow*width,
			d_temp + startRow*width,
			width, rows, d_gaussian, gaussianSize);

		// 3 Sobel
		sharedMemSize = (TILE_SIZE + sobelSize - 1) * (TILE_SIZE + sobelSize - 1) * sizeof(float);
		convolutionGeneric<<<grid, block, sharedMemSize, s>>>(
			d_temp + startRow*width,
			d_temp + startRow*width,
			width, rows, d_sobel, sobelSize);

		// 4 Threshold
		float threshold = 0.5f;
		threshold_kernel<<<grid, block, 0, s>>>(
			d_temp + startRow*width,
			d_output + startRow*width,
			width, rows, threshold);

		// 5 Copier le resultat Device -> Hote
		cudaMemcpyAsync(h_output + startRow * width,
						d_output + startRow * width,
						rows * width * sizeof(float),
						cudaMemcpyDeviceToHost, s);
	}

	cudaDeviceSynchronize();

	// ---------------- Affichage console ----------------
	cout << "    \033[36m\033[3mPipeline & Overlapping : avant\033[0m" << endl;
	for (int y = 0; y < 10; ++y) {
		for (int x = 0; x < 10; ++x)
			cout << fixed << setprecision(1) << h_input[y * width + x] << " ";
		cout << endl;
	}
	
	cout << "    \033[36m\033[3mPipeline & Overlapping : apres\033[0m" << endl;
	for (int y = 0; y < 10; ++y) {
		for (int x = 0; x < 10; ++x)
			cout << fixed << setprecision(1) << h_output[y * width + x] << " ";
		cout << endl;
	}

	// ---------------- Nettoyage ----------------
	cudaStreamDestroy(streams[0]); cudaStreamDestroy(streams[1]);
	cudaFree(d_input); cudaFree(d_output); cudaFree(d_temp);
	cudaFree(d_gaussian); cudaFree(d_sobel);
	delete[] h_input; delete[] h_output;
}