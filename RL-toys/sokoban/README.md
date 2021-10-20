# Sokoban

[Sokoban](https://en.wikipedia.org/wiki/Sokoban) is a game in which player controls a worker to push boxes to specific locations. 

## State Space
Each game here is surrounded with walls that could not be pushed. Player and boxes can move to the corridor inside the walls, length of which is noted by $n$. For $k$ boxes, and one player, there are $n\left(\begin{matrix}n-1 \\ k\end{matrix}\right)$ cases of position for these objects. So for small games, there are at most thousands of states. We would like to use dynamic programming to solve Bellman equation, i.e. using policy iteration and value iteration.

## Action Space
Player can only move up, down, left and right, thus action space is `Base.OneTo(4)`.


## Create Game
We define some characters for representing units in sokoban game:

- X: Wall
- O: Box
- *: Goal
- P: Player
- ` `: Empty space for corridors
- ⦿: Box on Goal(`\circledbullet<tab>`)
- @: Player on Goal

One can define its game such as `game/game1.txt`:

```
XXXXXXXXX
X       X
X*   P* X
XO      X
X    O  X
X       X
XXXXXXXXX
```


to play with it, run

```julia
julia> include("sokoban.jl")

julia> game = from_file("games/game1")

julia> interact!(game)

```

and type 'w', 's', 'a', 'd' then enter to execute each step. Type 'q' and enter to quit the game.


## Train RL Agent

To create a game with agent:

```julia
julia> include("agent.jl")

julia> game = new_game("games/game0")
```

To evalutate policy, run:
```julia
julia> sweep(game, 10)  # sweep 10 times for iteratively updating Vs
```

After policy evaluating, just let the agent play to show the preformance:
```julia
julia> replay(game)
```
