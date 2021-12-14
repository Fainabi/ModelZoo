using ReinforcementLearning
using PyCall


mutable struct CartPoleEnv <: AbstractEnv
    env
    reward
    terminated
    function CartPoleEnv()
        gym = pyimport("gym")
        env = gym.make("CartPole-v1")
        env.reset()
        new(env, 0, false)
    end
end

function RLBase.reset!(env::CartPoleEnv)
    env.env.reset()
    env.terminated = false
    env.reward = 0
end

RLBase.state_space(::CartPoleEnv) = WorldSpace()
RLBase.state(env::CartPoleEnv) = collect(env.env.env.state)

RLBase.action_space(::CartPoleEnv) = Base.OneTo(2)
RLBase.reward(env::CartPoleEnv) = env.reward

RLBase.is_terminated(env::CartPoleEnv) = env.terminated

function (env::CartPoleEnv)(action)
    observation, reward, done, info = env.env.step(action - 1)

    env.reward = reward
    env.terminated = done
end

render(env::CartPoleEnv) = env.env.render()
close(env::CartPoleEnv) = env.env.close()