// ============================================================================
//  FILENAME   : functionCPU.hpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-09
//  UPDATED    : 2025-10-10
//  DESCRIPTION: Step 5 Cuda - methods declaration
// ============================================================================

#pragma once

#include <iostream>
using namespace std;

#include <bits/stdc++.h>
#include <cuda_runtime.h>

void advancedReduction();
void scan();
void histogramme();
void pipeline();
void multiKernelPipeline();
