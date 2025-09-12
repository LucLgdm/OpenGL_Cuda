// ============================================================================
//  FILENAME   : Makefile
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-12
//  UPDATED    : 2025-09-12
//  DESCRIPTION: Makefile for Vector addition CUDA
// ============================================================================

# Compiler
NVCC = nvcc

# Flags (tu peux ajouter -G pour le debug, -O2 pour l'optimisation)
FLAGS = -arch=sm_75

# Source files
SRC = main.cu

# Output executable
NAME = hello_cuda

all: $(NAME)

$(NAME): $(SRC)
	$(NVCC) $(FLAGS) $^ -o $@

clean:
	rm -f $(NAME)

re: clean all

.PHONY: all clean re