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
julia> include("Sokoban.jl")

julia> game = from_file("games/game4")

julia> interact!(game)

```

and type 'w', 's', 'a', 'd' then enter to execute each action. Type 'q' and enter to quit the game.


## Train RL Agent

### Dynamic Programming
To create a game with agent:

```julia
julia> include("Sokoban.jl")

julia> game = new_game("games/game0")
```

To evaluate policy, run:
```julia
julia> dp = DP(game)

julia> sweep(dp, 10)  # sweep 10 times for iteratively updating Vs
```

After policy evaluating, just let the agent play to show the performance:
```julia
julia> replay(dp)
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
to the agent, here we set to give positive reward when it reaches true end. Note that
the rewards are still generated in-action.


### Monte-Carlo 
#### On Policy MC

For a game with relative big state space, e.g. game5, it needs time to travel all states in every single sweep.
It took a decade more sweeps to finish its playing. For game with bigger state space, consider monte-carlo and TD methods. 

```julia
julia> game = new_game("games/game1")

julia> mc = MC(game)

julia> run(mc)
```

To construct agent with action-value function, add `:Q` parameter:
```julia
julia> mc = MC(game, :Q)
```

One can also set type of first-visit MC or every-visit MC:

```julia
julia> mc = MC(game, :Q; kind=EVERY_VISIT)
```

The default number of episodes is 100, and hook is `OptimalTrajectoryHook`, which record the best trajectory in training
monte-carlo agent. Thus the command above is equal to

```julia
julia> run(mc, 100; hook=OptimalTrajectoryHook())
```

To replay the playing that agent took, run

```julia
julia> hook = run(mc)

julia> replay(mc, hook)
```

or let it play again following its policy

```julia
julia> replay(mc)
```

These replay can be quit by typing 'q' without entering it.

Agent would always take another action if the game is end, matching the sequence of

$$ S_{t-1}, A_{t-1}; R_t, S_t, A_t. $$

The small sokoban games can train agent very fast, while relative big ones need more episodes. 

**Note:** The rewards for every step are supposed to be negative, to push monte-carlo agent to explorer
new ways. Since the player can just walk up and down, without moving any of the boxes, and thus create 
the infinite steps of episode, we have set a boundary for it. And if agent take steps more than that, 
the game ends. If the game rushed into a dead state, the game ended and gave a punishment reward.

Such punishment shall less or equal to maximum step number times every-step reward, or the agent may learn
to go to these states to create an early end.


In the Sokoban game, if the agent succeeds one single time, it learns rapidly to create solutions. 
However, we initiate the state-value function with zeros, leaving no prior knowledge to agents. 
In such a scenery, agents can only explore and explore until they come across the true ends. 
To speed up the training, one can first make several DP steps, forming a rough preview of value functions,
and then perform monte-carlo methods.

```julia
julia> game = new_game("games/game5"; step_reward=-1.0)

julia> mc = MC(game)

julia> sweep(mc, 10)  # value iteration

julia> run(mc, 1000)  # monte-carlo prediction
```

#### Off Policy MC

To construct an off-policy MC agent, run:

```julia
julia> game = new_game("games/game2"; step_reward=-1.0)

julia> mc = MC(game; sampling=ORDINARY_IMPORTANCE_SAMPLING)
```

or

```julia
julia> mc = MC(game; sampling=WEIGHTED_IMPORTANCE_SAMPLING)
```

Note that our behaviour policy is a random policy while target policy is greedy, 
and thus it needs more and more episodes for target policy to learn from the random policy. 
It is recommanded to play with game0 to game2, which have smaller state space.





