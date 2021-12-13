# Cart Pole

The cart pole env is from openai's gym. The official [code](https://gym.openai.com/) is like

```python
import gym
env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
```

and with `PyCall`, we have 

```julia
using PyCall

gym = pyimport("gym")

env = gym.make("CartPole-v1")
observation = env.reset()
for _ in 1:1000
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done
        observation = env.reset()
    end
end

env.close()
```

Here we use [`ReinforcementLearning.jl`](https://juliareinforcementlearning.org/) and construct a wrapper for the envrionment. To use it, run:

```julia
include("cartpole.jl")

env = CartPoleEnv()
```

and use the native framework of `ReinforcementLearning.jl`:

```julia
policy = RandomPolicy()

run(policy, env, StopAfterStep(1000), RenderHook())
```

One can see the similar display of playing. 
