## ============================================================================
##  FILENAME   : Makefile
##  PROJECT    : GPU Rendering Playground
##  AUTHOR     : Luc <lucdemercey@gmail.com>
##  CREATED    : 2025-09-18
##  UPDATED    : 2025-09-11
##  DESCRIPTION: Makefile
## ============================================================================

# Compiler
CPP      = g++
FLAGS    = -MMD -std=c++17 -I include/
LDFLAGS  = -lglfw -ldl -lm -lGL

# Sources
vpath %.cpp src/
vpath %.c src/

SRC      = main.cpp Render.cpp Shapes.cpp Shader.cpp
SRCC     = glad.c

# Object directories
OBJDIR   = objs
OBJ      = $(SRC:%.cpp=$(OBJDIR)/%.o)
OBJGLAD  = $(OBJDIR)/glad.o
DEP      = $(OBJ:.o=.d)

# Target executable
NAME     = Lighting

# Default target
all: $(NAME)

# Link
$(NAME): $(OBJ) $(OBJGLAD)
	@$(CPP) $(OBJ) $(OBJGLAD) $(LDFLAGS) -o $@
	@echo "\033[32m$(NAME) created!\033[0m"

# Compile C++ sources
$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@$(CPP) $(FLAGS) -c $< -o $@

# Compile GLAD (C source)
$(OBJDIR)/glad.o: $(SRCC)
	@mkdir -p $(@D)
	@$(CPP) $(FLAGS) -c $< -o $@

# Clean object files
clean:
	@rm -rf $(OBJDIR)
	@echo "\033[34mDeleted object files!\033[0m"

# Clean object files + executable
fclean: clean
	@rm -f $(NAME)
	@echo "\033[35mDeleted everything!\033[0m"

# Rebuild
re: fclean all
	@echo "\033[33mRebuild done!\033[0m"

# Include dependencies
-include $(DEP)

# Phony targets
.PHONY: all clean fclean re
