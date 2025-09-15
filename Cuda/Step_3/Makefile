// ============================================================================
//  FILENAME   : Makefile
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-13
//  UPDATED    : 2025-09-12
//  DESCRIPTION: Makefile for 2D/3D Grids & Memory Hierarchy
// ============================================================================

# Compiler
NVCC = nvcc

# Flags (tu peux ajouter -G pour le debug, -O2 pour l'optimisation)
FLAGS = -arch=sm_50

# Source files
SRC = main.cu

# Output executable
NAME = Matrix

all: $(NAME)

$(NAME): $(SRC)
	$(NVCC) $(FLAGS) $^ -o $@

clean:
	rm -f $(NAME)

re: clean all

.PHONY: all clean re