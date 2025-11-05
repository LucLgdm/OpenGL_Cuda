// ============================================================================
//  FILENAME   : main.cu
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-29
//  UPDATED    : 2025-11-05
//  DESCRIPTION: Fusion : Step 1 - Ripple effect
// ============================================================================

#include "../includes/kernel.cuh"

__global__ void rippleKernel(uchar4* pixels, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float fx = x - width/2.0f;
    float fy = y - height/2.0f;
    float dist = sqrtf(fx*fx + fy*fy);
    unsigned char color = (unsigned char)(128.0f + 127.0f * cosf(dist/10.0f - time));
    pixels[y*width + x] = make_uchar4(color, color, 255, 255);
}