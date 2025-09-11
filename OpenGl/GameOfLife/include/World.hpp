/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   World.hpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: lde-merc <lde-merc@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/09/11 09:39:47 by lde-merc          #+#    #+#             */
/*   Updated: 2025/09/11 11:36:35 by lde-merc         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#pragma once

#include <iostream>
using namespace std;

#include <vector>

#include "Grid.hpp"

class World {
	public:
		World();
		~World();
		
		const Grid& grid() const { return _grid; }
		void step();		
		
	private:
		Grid	_grid;
};
