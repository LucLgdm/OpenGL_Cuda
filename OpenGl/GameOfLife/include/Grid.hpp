/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Grid.hpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: lde-merc <lde-merc@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/09/11 09:42:26 by lde-merc          #+#    #+#             */
/*   Updated: 2025/09/11 10:46:26 by lde-merc         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#pragma once

#include <iostream>
using namespace std;

#include <unordered_set>
#include <utility>   // for std::pair
#include <functional> // for std::hash


struct PairHash {
	size_t operator()(const pair<int,int>& p) const {
		return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
	}
};

class Grid {
	public:
		Grid();
		~Grid();
		
		void setAlive(int x, int y);
		bool isAlive(int x, int y) const;
		const unordered_set<pair<int,int>, PairHash>& getAliveCells() const { return _aliveCells; }
		int countNeighbors(int x, int y) const;
		void update(const unordered_set<pair<int, int>, PairHash>& nextGen) { _aliveCells = nextGen; }
		Grid &operator=(const Grid &other) = default;

	private:
		unordered_set<pair<int, int>, PairHash>	_aliveCells;
};
