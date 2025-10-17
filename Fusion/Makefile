## ============================================================================
##  FILENAME   : Makefile
##  PROJECT    : GPU Rendering Playground
##  AUTHOR     : Luc <lucdemercey@gmail.com>
##  CREATED    : 2025-10-17
##  UPDATED    : 2025-09-11
##  DESCRIPTION: Makefile
## ============================================================================

# === Compilateurs ===
CXX  = g++
NVCC = nvcc

# === Options ===
CXXFLAGS = -MMD -std=c++17 -O2 \
	-Iexternal/glad -Iexternal/KHR -Iexternal/glm -Iexternal/stb
NVCCFLAGS = -std=c++17 -O2 \
	-Iexternal/glad -Iexternal/KHR -Iexternal/glm -Iexternal/stb
LDFLAGS = -lGL -lglfw -ldl -lcuda

# Prevent recursive make from printing "Entering/Leaving directory" lines
MAKEFLAGS += --no-print-directory
export MAKEFLAGS

# === Fichiers communs ===
COMMON_SRC = external/glad/glad.c
COMMON_OBJ = $(COMMON_SRC:.c=.o)

# === Steps ===
STEPS = Step_1 Step_2 Step_3
BINARIES = $(STEPS:%=bin/%)

# === Fichiers sources pour chaque step ===
define make-step-vars
$(1)_CPP_SRC := $(wildcard $(1)/srcs/*.cpp)
$(1)_CU_SRC  := $(wildcard $(1)/cuda/*.cu)
$(1)_OBJ     := $$($(1)_CPP_SRC:.cpp=.o) $$($(1)_CU_SRC:.cu=.o)
endef
$(foreach step,$(STEPS),$(eval $(call make-step-vars,$(step))))

# === Règle principale ===
all: $(BINARIES)

# === Compilation d’un step (fusionne .cpp + .cu + communs) ===
bin/%: $(COMMON_OBJ)
	@mkdir -p bin
	@if [ -f $*/srcs/main.cpp ]; then \
		echo "\033[33mBuilding C++ step $*...\033[0m"; \
		$(CXX) $(CXXFLAGS) -I$*/includes -o $@ $*/srcs/main.cpp $(COMMON_OBJ) $(LDFLAGS); \
	elif [ -f $*/srcs/main.cu ]; then \
		echo "\033[33mBuilding CUDA step $*...\033[0m"; \
		$(NVCC) -O2 -std=c++17 -Iexternal/glad -Iexternal/KHR -Iexternal/glm -Iexternal/stb \
			-I$*/includes -o $@ $*/srcs/main.cu $(COMMON_OBJ) -lGL -lglfw -ldl -lcuda; \
	else \
		@echo "\033[35mNo main.cpp or main.cu found for $*\033[0m"; exit 1; \
	fi
	@echo "\033[32m$@ is created\033[0m"


# === Compilation générique des fichiers ===
%.o: %.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@

external/glad/glad.o: external/glad/glad.c
	@$(CXX) -c $< -o $@ -Iexternal/glad -Iexternal/KHR -Iexternal/glm -Iexternal/stb

# === Règle pratique pour exécuter un step ===
run:
	@step=$${STEP:-Step_1}; \
	echo "\033[34mRunning $$step...\033[0m"; \
	$(MAKE) bin/$$step && ./bin/$$step

# === Nettoyage ===
clean:
	@find . -name "*.o" -delete
	@find . -name "*.d" -delete
	@echo "\033[34mDeleted object files!\033[0m"

fclean: clean
	@rm -rf bin/
	@echo "\033[35mDeleted everything!\033[0m"

re: fclean all
	@echo "\033[33mRebuild done!\033[0m"

# === Dépendances ===
-include $(shell find . -name "*.d" 2>/dev/null)

.PHONY: all clean fclean re