# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: lde-merc <lde-merc@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/10 16:05:16 by lde-merc          #+#    #+#              #
#    Updated: 2025/09/11 11:42:56 by lde-merc         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

vpath %.cpp src/
vpath %.c src/

NAME = LifeGame
FLAGS = -MMD -lglfw -ldl -lm -lGL -I include/
CPP = g++

SRC = main.cpp Grid.cpp World.cpp
SRCC = glad.c
OBJDIR = objs
OBJ = $(SRC:%.cpp=$(OBJDIR)/%.o)
OBJGLAD = $(OBJDIR)/glad.o
DEP		:= $(OBJ:.o=.d)

all: $(NAME)

$(NAME): $(OBJ) $(OBJGLAD)
	@$(CPP) $(OBJ) $(OBJGLAD) $(FLAGS) -o $(NAME)
	@echo "\033[32m$(NAME) created !\033[0m"

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@$(CPP) $(FLAGS) -c $< -o $@

$(OBJDIR)/glad.o: $(SRCC)
	@mkdir -p $(@D)
	@$(CPP) $(FLAGS) -c $< -o $@


clean:
	@rm -rf $(OBJDIR)
	@echo "\033[34mDeleting almost everything !\033[0m"
	
fclean: clean
	@rm -f $(NAME)
	@echo "\033[35mDeleting everything !\033[0m"
	
re: fclean all
	@echo "\033[33mmake re SUCCESS!\033[0m"

-include $(DEP)

.PHONY: all clean fclean re