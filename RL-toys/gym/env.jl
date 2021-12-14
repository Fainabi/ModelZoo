using ReinforcementLearning
using PyCall


mutable struct GymEnv <: AbstractEnv
    env
    reward
    terminated
    function GymEnv(name)
        gym = pyimport("gym")
        env = gym.make(name)
        env.reset()
        new(env, 0, false)
    end
end

function RLBase.reset!(env::GymEnv)
    env.env.reset()
    env.terminated = false
    env.reward = 0
end

RLBase.state_space(::GymEnv) = WorldSpace()
RLBase.state(env::GymEnv) = collect(env.env.env.state)

RLBase.action_space(env::GymEnv) = if hasproperty(env.env.action_space, :n)
    Base.OneTo(env.env.action_space.n)
else
    RLEnvs.ClosedInterval(env.env.action_space.low, env.env.action_space.high)
end

RLBase.reward(env::GymEnv) = env.reward

RLBase.is_terminated(env::GymEnv) = env.terminated

function (env::GymEnv)(action)
    observation, reward, done, info = env.env.step(action - 1)

    env.reward = reward
    env.terminated = done
end

render(env::GymEnv) = env.env.render()
close(env::GymEnv) = env.env.close()
