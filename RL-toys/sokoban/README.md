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

One can define its game such as `games/game4`:

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

To evaluate policy, run:
```julia
julia> sweep(game, 10)  # sweep 10 times for iteratively updating Vs
```

After policy evaluating, just let the agent play to show the performance:
```julia
julia> replay(game)
```

We choose random policy here, and does not improve the policy, so sweep there
take $\gamma = 1$, and as a result, we would not take value iteration.

In the book of `Reinforcement Learning: An introduction`
the reward was in the sequence like:
$$
s_{t-1}, a_{t-1}; r_t, s_t, a_t; \cdots 
$$

and some examples assign 0 to the termination state, -1 to others.
In sokoban, there exists many states that are dead states, in which 
we can never reach the true solution in such state. So rather than give penalty
to the agent, here we set to give positive reward when it reaches true end. And 
thus the sequence is with in-time rewards
$$
s_{t-1}, a_{t-1}, r_{t-1}; s_t, a_t, r_t; \cdots 
$$

another reason to take such approach is the implementation of policy iteration
in ReinforcementLearning.jl, where the V(t+1) is factored with zero when meeting
termination condition.


For a game with relative big state space, e.g. game5, it needs time to travel all states in every single sweep.
It took a decade more sweeps to finish its playing.
