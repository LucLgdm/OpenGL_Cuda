# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: lde-merc <lde-merc@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/10 16:05:16 by lde-merc          #+#    #+#              #
#    Updated: 2025/09/10 16:05:47 by lde-merc         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

NAME = LifeGame
FLAGS = -MMD -Wall -Wextra -Werror -O3 -std=c++98
CPP = c++

SRC = main.cpp 
OBJDIR = objs
OBJ = $(SRC:%.cpp=$(OBJDIR)/%.o)
DEP		:= $(OBJ:.o=.d)

all: $(NAME)

$(NAME): $(OBJ)
	@$(CPP) $(FLAGS) $(OBJ) -o $(NAME)
	@echo "\033[32m$(NAME) created !\033[0m"
	
$(OBJDIR)/%.o: %.cpp
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