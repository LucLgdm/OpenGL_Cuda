// ============================================================================
//  FILENAME   : main.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-12
//  UPDATED    : 2025-09-12
//  DESCRIPTION: Step 1 Cuda - Hello CUDA
// ============================================================================

#include <cuda_runtime.h>

/*****************************************************
 * A thread is a little task exec uted on a Cuda Core
 * A Block is a group of thread (1D, 2D or 3D)
 * A Grid is a group of Block
 *****************************************************/

/*************************************************************************
 * __global__ is a function called from the CPU, executed in the GPU
 * __device__ is a function called from a kernel, executed only in a GPU
 * __host__   is a CPU function (implicite) 
 *************************************************************************/

 // Kernel CUDA
__global__ void helloFromGPU() {
	int threadId = threadIdx.x;   // ID dans le bloc
	int blockId  = blockIdx.x;    // ID du bloc
	int globalId = blockIdx.x * blockDim.x + threadIdx.x; // ID global
	/*****************************************************
	 * printf is used because, std::cout and std::vector
	 * don't work in a kernel
	 *****************************************************/
	printf("Hello from thread %d in block %d (global %d)\n",
		threadId, blockId, globalId);
}

int main() {
	// 2 block of 4 threads are involved
	helloFromGPU<<<2, 4>>>();
	cudaDeviceSynchronize(); // wait for the end of the kernel
	return 0;
}
