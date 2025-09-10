# Life_game

This project is a C++ implementation of John Horton Conway's famous *Game of Life*, a cellular automaton.

The grid is divided into cells, and each cell can be either alive or dead.  
A neighbor is defined as any cell that shares an edge or a corner, meaning each cell has up to 8 neighbors.

The evolution of the grid is governed by the following rules:

- If a cell is dead and has exactly 3 living neighbors, it becomes alive.  
- If a cell is alive and has 2 or 3 living neighbors, it remains alive; otherwise, it dies.  

At each step, all cells are updated simultaneously according to these rules.

Although the rules are simple, they can generate surprisingly complex behavior.

https://codex.forge.apps.education.fr/exercices/jeu_de_la_vie/

For the *Day & Night* game, the rules are almost the same :

- If a cell is dead and surounded by 3, 6, 7 or 8 neighbors alive, the cell lives.
- If a cell is alive and has 3, 4, 6, 7 or 8 neighbors alive, the cell lives.

Same as above for the *Highlight* game :

- If a dead cell is surrounded by 3 or 6 living cells, the cell lives.
- If a living cell is surrounded by 2 or 3 living cell, the cell lives.

