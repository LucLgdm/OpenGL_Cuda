// ============================================================================
//  FILENAME   : main.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-09
//  UPDATED    : 2025-10-15
//  DESCRIPTION: Step 5 Cuda - Going further
// ============================================================================

#include "functionCPU.hpp"

int main() {
	cout << "\033[34m----------Advanced multi-step reduction----------\033[0m" << endl;
	advancedReduction();
	cout << endl << "\033[34m----------Scan (Prefix Sum)----------\033[0m" << endl;
	scan();
	cout << endl << "\033[34m----------Histogramme----------\033[0m" << endl;
	histogramme();
	cout << endl << "\033[34m----------Pipeline convolution----------\033[0m" << endl;
	pipeline();
	cout << endl << "\033[34m----------Multikernel pipeline----------\033[0m" << endl;
	multiKernelPipeline();
	cout << endl << "\033[34m----------Max value array----------\033[0m" << endl;
	maxValue();
	return 0;
}
