using ReinforcementLearning
using Random


mutable struct KRandomWalkEnv <: AbstractEnv
    true_values
    true_reward
    reward::Float32 
    is_terminated::Bool 
    rng::AbstractRNG
    KRandomWalkEnv(k=10, true_reward=0.) = new(zeros(k) .+ true_reward, true_reward, 0., false, Random.GLOBAL_RNG)
end



RLBase.action_space(env::KRandomWalkEnv) = Base.OneTo(length(env.true_values))
RLBase.is_terminated(env::KRandomWalkEnv) = env.is_terminated
RLBase.state(env::KRandomWalkEnv) = 1
RLBase.state_space(_env::KRandomWalkEnv) = Base.OneTo(1)
RLBase.reward(env::KRandomWalkEnv) = env.reward
function RLBase.reset!(env::KRandomWalkEnv) 
    env.true_values *= 0
    env.is_terminated = false
end

function (env::KRandomWalkEnv)(action)
    env.reward = randn(env.rng) + env.true_values[action]
    # start random walk
    env.true_values .+= randn!(env.rng, similar(env.true_values)) / 10  # make var = 0.01
    env.is_terminated = true
end

Random.seed!(env::KRandomWalkEnv, x) = seed!(env.rng, x)


RLBase.RewardStyle(::MultiArmBanditsEnv) = TERMINAL_REWARD
