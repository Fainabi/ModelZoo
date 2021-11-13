using ReinforcementLearning

Base.@kwdef mutable struct MonteCarloTreeSearchPolicy <: AbstractPolicy
    learner::AbstractLearner
    explorer::AbstractExplorer
    episodes_per_search::Int = 100
    search_mode::Bool = true
end

function (π::MonteCarloTreeSearchPolicy)(env; search_mode=true)
    if search_mode
        for _ in 1:π.episodes_per_search
            select!(π, env)
            expand!(π, env)
            simulate!(π, env)
            backup!(π, env)
        end
    end

    π(env, ActionStyle(env), action_space(env))
end
(π::MonteCarloTreeSearchPolicy)(env) = π(env, ActionStyle(env), action_space(env))
(π::MonteCarloTreeSearchPolicy)(env, ::MinimalActionSet, ::Base.OneTo) = π.explorer(π.learner(env))
(π::MonteCarloTreeSearchPolicy)(env, ::FullActionSet, ::Base.OneTo) = 
    π.explorer(π.learner(env), legal_action_space_mask(env))

# search algorithm does not need to approxiate the values
function RLBase.update!(::MonteCarloTreeSearchPolicy, ::AbstractTrajectory) end


# implementation

function select!(p::MonteCarloTreeSearchPolicy, env::AbstractEnv)
    
end

function expand!(p::MonteCarloTreeSearchPolicy, env::AbstractEnv)

end

function simulate!(p::MonteCarloTreeSearchPolicy, env::AbstractEnv)
    
end

function backup!(p::MonteCarloTreeSearchPolicy, env::AbstractEnv)
    
end

