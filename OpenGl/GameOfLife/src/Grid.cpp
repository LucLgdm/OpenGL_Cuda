/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Grid.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: lde-merc <lde-merc@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/09/11 10:23:32 by lde-merc          #+#    #+#             */
/*   Updated: 2025/09/11 11:37:39 by lde-merc         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Grid.hpp"

Grid::Grid() {}
Grid::~Grid() {}

void Grid::setAlive(int x, int y) {
	_aliveCells.insert({x, y});
}

bool Grid::isAlive(int x, int y) const {
	return _aliveCells.find({x, y}) != _aliveCells.end();
}

int Grid::countNeighbors(int x, int y) const {
	int count = 0;
	for (int dx = -1; dx <= 1; ++dx) {
		for (int dy = -1; dy <= 1; ++dy) {
			if (dx == 0 && dy == 0) continue; // Skip the cell itself
			if (isAlive(x + dx, y + dy)) {
				++count;
			}
		}
	}
	return count;
}
