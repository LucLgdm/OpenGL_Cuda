/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   World.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: lde-merc <lde-merc@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/09/11 10:32:35 by lde-merc          #+#    #+#             */
/*   Updated: 2025/11/04 15:04:35 by lde-merc         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "World.hpp"

void parseRLE(const std::string& rle, Grid& grid, int x0 = 0, int y0 = 0) {
    int x = 0;
    int y = 0;
    size_t i = 0;
    int num = 0;

    while (i < rle.size()) {
        char c = rle[i];

        if (std::isdigit(c)) {
            num = num * 10 + (c - '0');
        } else if (c == 'b') {
            if (num == 0) num = 1;
            x += num;
            num = 0;
        } else if (c == 'o') {
            if (num == 0) num = 1;
            for (int j = 0; j < num; ++j) {
                grid.setAlive(x0 + x, y0 + y);
                x++;
            }
            num = 0;
        } else if (c == '$') {
            if (num == 0) num = 1;
            y += num;
            x = 0;
            num = 0;
        } else if (c == '!') {
            break;
        }
        i++;
    }
}


// Constructeur
World::World() {
	string pattern;
	
	pattern =
		"4b2o6b2o4b$3bobo6bobo3b$3bo10bo3b$2obo10bob2o$2obobo2b2o2bobob2o$3bobo"
		"bo2bobobo3b$3bobobo2bobobo3b$2obobo2b2o2bobob2o$2obo10bob2o$3bo10bo3b$"
		"3bobo6bobo3b$4b2o6b2o!";
	parseRLE(pattern, _grid, 0, 0);

	// pattern = 
	// 	"23bo$22bobo$21bobobo$21bo3bo$13b2o4b2ob2ob2o$13bo2bobobo3bo$15b2obo3bo"
	// 	"bo$16bobo2b2ob2o$4bo10b2o3bo2bo$4b3o4b2o4b2ob2obo$7bo3b2o6bo2bo$6b2o"
	// 	"13bo$22b3o$18b2o4bo$2b2o14bo$2bo13bobo$4bo11b2o$3b2o6bo$2bo6b3o$bob2o"
	// 	"4bobo$o2bobo5bo$2o2bo2$15b2o$8b2o5bobo$9bo7bo$6b3o8b2o$6bo!";
	// parseRLE(pattern, _grid, -25, -25);

	// pattern = 
	// 	"17b2o$5bo11bo$5b3o6b2obo$8bo4bo2bo$7b2o4b2o45bo7bo$59bobo5bobo$28b2o"
	// 	"30bo7bo$22b2o5bo$22bobob3o$24bobo$23bobo6b2o21bo17bo$24bo7bo22bo17bo$"
	// 	"10bo19bobo22bo5bo5bo5bo$9b3o18b2o19b2o7b3o3b3o7b2o$8bo3bo30b2o5bobo6b"
	// 	"2ob2ob2ob2o6bobo$8b2ob2o30bo6bo9b3o3b3o9bo$21b2ob2o15bobo5b2o10bo5bo"
	// 	"10b2o$21bo3bo15b2o$2b2o18b3o31b2o13b2o$bobo19bo13bo18b2o13b2o$bo7bo26b"
	// 	"obo$2o6bobo24bo3bo9b2o10bo5bo10b2o$7bobo26bobo11bo9b3o3b3o9bo$5b3obobo"
	// 	"25bo12bobo6b2ob2ob2ob2o6bobo$4bo5b2o39b2o7b3o3b3o7b2o$4b2o28b2o19bo5bo"
	// 	"5bo5bo$35bo19bo17bo$19b2o4b2o5b3o20bo17bo$17bo2bo4bo6bo$16bob2o6b3o$"
	// 	"16bo11bo16b2o$15b2o27bo2bo12bo7bo$44b3o12bobo5bobo$60bo7bo4$42b2o$43bo"
	// 	"$40b3o5b2o$40bo7b2o!";
	// parseRLE(pattern, _grid, -40, -20);

	// pattern = 
	// 	"11bo38b$10b2o38b$9b2o39b$10b2o2b2o34b$38bo11b$38b2o8b2o$39b2o7b2o$10b"
	// 	"2o2b2o18b2o2b2o10b$2o7b2o39b$2o8b2o38b$11bo38b$34b2o2b2o10b$39b2o9b$"
	// 	"38b2o10b$38bo!";
	// parseRLE(pattern, _grid, -25, -7);

	// pattern = 
	// 	"bob$2bo$3o!";
	// parseRLE(pattern, _grid, 0, 0);

	// pattern = 
	// 	"10b2o$10bobo$13bo2b2o$11b2obo2bo$10bobob2o$11bo$21b2o$21bo$6bo5b3o4bobo$5bobo3bo3bo3b2o4bo$5b2o3bo5bo7bobo$3b2o5bo5bo7bobo$2bo2b2o3bo5bo6b2ob3o$bobo2bo4bo3bo13bo$o2b2o7b3o8b2ob3o$2o21b2obo$6bob2o21b2o$4b3ob2o8b3o7b2o2bo$3bo13bo3bo4bo2bobo$4b3ob2o6bo5bo3b2o2bo$6bobo7bo5bo5b2o$6bobo7bo5bo3b2o$7bo4b2o3bo3bo3bobo$11bobo4b3o5bo$11bo$10b2o$21bo$17b2obobo$15bo2bob2o$15b2o2bo$20bobo$21b2o!";
	// parseRLE(pattern, _grid, -40, -40);
}

World::~World() {}

void World::step() {
	unordered_set<pair<int, int>, PairHash> candidates;
	unordered_set<pair<int, int>, PairHash> nextGen;

	unordered_set<pair<int, int>, PairHash> current = _grid.getAliveCells();

	for (auto& cell : current) {
		int x = cell.first, y = cell.second;

		// Adding cell all neighbors to candidates
		for (int dx = -1; dx <= 1; ++dx) {
			for (int dy = -1; dy <= 1; ++dy) {
				candidates.insert({x + dx, y + dy});
			}
		}
	}

	// Apply rules
	for (auto& cell : candidates) {
		int x = cell.first, y = cell.second;
		int neighbors = _grid.countNeighbors(x, y);
		if (neighbors == 3 || (neighbors == 2 && _grid.isAlive(x, y))) {
			nextGen.insert({x, y});
		}
	}
	_grid.update(nextGen);
}
