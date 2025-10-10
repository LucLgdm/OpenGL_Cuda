// ============================================================================
//  FILENAME   : main.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-09
//  UPDATED    : 2025-10-10
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
	cout << endl << "\033[34m----------Multikernel Pipeline----------\033[0m" << endl;
	multiKernelPipeline();
	return 0;
}
