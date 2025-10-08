// ============================================================================
//  FILENAME   : Makefile
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-13
//  UPDATED    : 2025-09-12
//  DESCRIPTION: Makefile for Matric operation with shared memory
// ============================================================================

# Compiler
NVCC = nvcc

# Flags (tu peux ajouter -G pour le debug, -O2 pour l'optimisation)
FLAGS = -arch=sm_50

# Source files
SRC = main.cu
SRC2 = main2D.cu

# Output executable
NAME = 1D
NAME2 = 2D

all: $(NAME) $(NAME2)

$(NAME): $(SRC)
	$(NVCC) $(FLAGS) $^ -o $@

$(NAME2): $(SRC2)
	$(NVCC) $(FLAGS) $^ -o $@

clean:
	rm -f $(NAME)
	rm -f $(NAME2)

re: clean all

.PHONY: all clean re