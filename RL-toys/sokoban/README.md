# Sokoban

[Sokoban](https://en.wikipedia.org/wiki/Sokoban) is a game in which player controls a worker to push boxes to specific locations. 

## State Space
Each game here is surrounded with walls that could not be pushed. Player and boxes can move to the corridor inside the walls, length of which is noted by $n$. For $k$ boxes, and one player, there are $n\left(\begin{matrix}n-1 \\ k\end{matrix}\right)$ cases of position for these objects. So for small games, there are at most thousands of states. We would like to use dynamic programming to solve Bellman equation, i.e. using policy iteration and value iteration.

## Action Space
Player can only move tu up, down, left and right, thus action space is `Base.OneTo(4)`.

