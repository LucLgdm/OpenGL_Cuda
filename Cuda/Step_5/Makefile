// ============================================================================
//  FILENAME   : Makefile
//  PROJECT    : GPU Computing with CUDA
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-09
//  UPDATED    : 2025-10-09
//  DESCRIPTION: Makefile for Step 5
// ============================================================================

# Compiler
NVCC = nvcc

# Flags (-G pour le debug, -O2 pour l'optimisation)
FLAGS = -arch=native
# FLAGS = -arch=sm_50

# Source files
SRC = main.cpp reduction.cu scan.cu histogramme.cu Convolution.cu multikernel.cu atomic_Warp.cu

# Output executable
NAME = Step_5

all: $(NAME)

$(NAME): $(SRC)
	@$(NVCC) $(FLAGS) $^ -o $@
	@echo "\033[32mBuild complete: $(NAME)\033[0m"

complete: $(NAME)
	@echo "\033[36mCheck memory errors with cuda-memcheck\033[0m"
	@cuda-memcheck ./$(NAME)
	@echo "\033[36mCheck the race conditions with cuda-memcheck --tool racecheck\033[0m"
	@cuda-memcheck --tool racecheck ./$(NAME)
	@echo "\033[36mCheck the leaks with cuda-memcheck --leak-check full\033[0m"
	@cuda-memcheck --leak-check full ./$(NAME)
	@echo "\033[36mCheck the synchronization errors with cuda-memcheck --tool synccheck\033[0m"
	@cuda-memcheck --tool synccheck ./$(NAME)
# 	@echo "\033[34mProfile the performance with Nsight Compute\033[0m"
# 	@nsys profile --stats=true ./$(NAME)

fclean:
	@rm -f $(NAME)
	@echo "\033[35mDelete $(NAME)\033[0m"

re: fclean all

.PHONY: all clean re