// ============================================================================
//  FILENAME   : particule.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-15
//  UPDATED    : 2025-10-16
//  DESCRIPTION: Step 5 Cuda - little particule simulation, velocity and force
// ============================================================================

#include "function.cuh"

__device__ inline float2 make_float2_device(float x, float y) {
	float2 r;
	r.x = x; r.y = y;
	return r;
}

__global__ void updateParticles(Particle* particles, int n, float g, float dt, float eps) {
	extern __shared__ Particle tile[];
	
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + tid;
	if (gid >= n) return;

	// Copie local de la particule courante
	Particle p = particles[gid];
	float2 acc = make_float2_device(0.0f, 0.0f);
	int tiles = (n + blockDim.x -1) / blockDim.x;

	for (int t = 0; t < tiles; ++t) {
		int idx = t * blockDim.x + tid;

		if (idx < n) tile[tid] = particles[idx];
		__syncthreads();

		int limit = (t == tiles - 1) ? (n - t * blockDim.x) : blockDim.x;

		// Calcul de l'acceleration
		for (int j = 0; j < limit; ++j) {
			// index global de la particule j de la tuile
			int gj = t * blockDim.x + j;
			if (gj == gid) continue; // n'interagit pas avec elle-meme

			float dx = tile[j].pos.x - p.pos.x;
			float dy = tile[j].pos.y - p.pos.y;
			float dist2 = dx * dx + dy * dy + eps * eps; // Softening
			float invDist = rsqrtf(dist2); // 1/r
			float invDist3 = invDist * invDist * invDist; // 1/r^3
			// a_i += G * m_j * r_vec / r^3
			float s = g * tile[j].m * invDist3;
			acc.x += s * dx;
			acc.y += s * dy;
		}
		__syncthreads();
	}

	// integration pour la vitesse et la position
	p.vel.x += acc.x * dt;
	p.vel.y += acc.y * dt;
	p.pos.x += p.vel.x * dt;
	p.pos.y += p.vel.y * dt;

	particles[gid] = p;
}

void particuleSystem() {
	const int BLOCK_SIZE = 128;
	const float g = 0.001f;  // constante gravitationnelle simplifiee
	const float dt = 0.1f;   // pas de temps
	const int n = 512;
	const float eps = 0.1f;
	vector<Particle> h_particles(n);

	for(int i = 0; i < n; i++) {
		h_particles[i].pos = make_float2((float)rand() / RAND_MAX * 100.0f, (float)rand() / RAND_MAX * 100.0f);
		h_particles[i].vel = make_float2(0.0f, 0.0f);
		h_particles[i].m = 1.0f + (float)rand() / RAND_MAX * 4.0f;
	}

	cout << "\033[33mBefore simulation :\033[0m" << endl;
	for(int i = 0; i < 10; ++i) {
		cout << "\033[32mParticule " << i << " :\033[0m" << endl
			<< "\033[32mPos =\033[0m (" << h_particles[i].pos.x << ", " << h_particles[i].pos.y << ")" << endl
			<< "\033[32mVel =\033[0m (" << h_particles[i].vel.x << ", " << h_particles[i].vel.y << ")" << endl;
	}

	Particle* d_particles;
	cudaMalloc(&d_particles, n * sizeof(Particle));
	cudaMemcpy(d_particles, h_particles.data(), n * sizeof(Particle), cudaMemcpyHostToDevice);

	dim3 threads(BLOCK_SIZE);
	dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

	for(int step = 0; step < 1000; ++step) {
		updateParticles<<<blocks, threads, BLOCK_SIZE * sizeof(Particle)>>>(d_particles, n, g, dt, eps);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(h_particles.data(), d_particles, n * sizeof(Particle), cudaMemcpyDeviceToHost);

	cout << "\033[33mAfter simulation :\033[0m" << endl;
	for(int i = 0; i < 10; ++i) {
		cout << "\033[32mParticule " << i << " :\033[0m" << endl
			<< "\033[32mPos =\033[0m (" << h_particles[i].pos.x << ", " << h_particles[i].pos.y << ")" << endl
			<< "\033[32mVel =\033[0m (" << h_particles[i].vel.x << ", " << h_particles[i].vel.y << ")" << endl;
	}

	cudaFree(d_particles);
}
