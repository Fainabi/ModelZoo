# Imitate the implementation of `MultiArmBanditsEnv` in `ReinforcementLearning.jl`
using ReinforcementLearning
using Random

mutable struct KArmedBanditsEnv <: AbstractEnv
    true_values
    is_terminated
    true_reward
    reward
    rng
end


function KArmedBanditsEnv(k=10; true_reward=0., rng=Random.GLOBAL_RNG)
    KArmedBanditsEnv(randn(k) .+ true_reward, false, true_reward, 0., rng)
end

RLBase.action_space(env::KArmedBanditsEnv) = Base.OneTo(length(env.true_values))
RLBase.is_terminated(env::KArmedBanditsEnv) = env.is_terminated
RLBase.reward(env::KArmedBanditsEnv) = env.reward
RLBase.state(env::KArmedBanditsEnv) = 1
RLBase.state_space(_env::KArmedBanditsEnv) = Base.OneTo(1)

# one time running
function (env::KArmedBanditsEnv)(action)
    env.reward = randn(env.rng) + env.true_values[action]
    env.is_terminated = true
end
function RLBase.reset!(env::KArmedBanditsEnv)
    env.is_terminated = false
end

Random.seed!(env::KArmedBanditsEnv, x) = seed!(env.rng, x)

# SINGLE_AGENT isa SingleAgent <: RLBase.AbstractNumAgentStyle
RLBase.NumAgentStyle(::MultiArmBanditsEnv) = SINGLE_AGENT

# Environment with the DynamicStyle of SEQUENTIAL must takes actions from different players one-by-one.
RLBase.DynamicStyle(::MultiArmBanditsEnv) = SEQUENTIAL

# All actions in the action space of the environment are legal
RLBase.ActionStyle(::MultiArmBanditsEnv) = MINIMAL_ACTION_SET

# the distribution of noise and original reward is unknown to the agent
RLBase.InformationStyle(::MultiArmBanditsEnv) = IMPERFECT_INFORMATION  

RLBase.StateStyle(::MultiArmBanditsEnv) = Observation{Int}()

# Only get reward at the end of environment
RLBase.RewardStyle(::MultiArmBanditsEnv) = TERMINAL_REWARD

RLBase.UtilityStyle(::MultiArmBanditsEnv) = GENERAL_SUM

# the same action lead to different reward each time.
# env needs an AbstractRNG, and seed!(env) needs implementing
RLBase.ChanceStyle(::MultiArmBanditsEnv) = STOCHASTIC  