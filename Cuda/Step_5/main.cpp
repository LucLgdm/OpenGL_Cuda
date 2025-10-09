// ============================================================================
//  FILENAME   : main.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-09
//  UPDATED    : 2025-10-09
//  DESCRIPTION: Step 5 Cuda - Going further
// ============================================================================

#include "function.cuh"

int main() {
	cout << "\033[34m----------Advanced multi-step reduction----------\033[0m" << endl;
	advancedReduction();
	cout << "\033[34m----------Scan (Prefix Sum)----------\033[0m" << endl;
	scan();
	return 0;
}
