// ============================================================================
//  FILENAME   : kernel.cuh
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-28
//  UPDATED    : 2025-11-05
//  DESCRIPTION: Fusion : Step 1 - Ripple effect
// ============================================================================

#pragma once

#include <iostream>
using namespace std;

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <cmath>

__global__ void rippleKernel(uchar4* pixels, int width, int height, float time);

