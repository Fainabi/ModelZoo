using ReinforcementLearning
using Flux

"""
    The Monte-Carlo Tree Approximator is a Value Based Approximator
"""

Base.@kwdef struct MonteCarloTreeApproximator{T, O} <: AbstractApproximator
    tree::T
    optimizer::O
end

function (app::MonteCarloTreeApproximator)(s)
    node = get(app.tree.id_map, s, nothing)
    if isnothing(node)
        0.0
    else
        reward(node)  # not @view, what if using a UCB explorer?
    end
end
function RLBase.update!(app::MonteCarloTreeApproximator, correction::Pair{Int,Float64})
    id, e = correction
    x = @view [app.tree(id).val][]
    x̄ = @view [e][]
    Flux.Optimise.update!(app.optimizer, x, x̄)
    app.tree(id).val = x[]
end


Base.@kwdef struct MonteCarloTreeLearner{A} <: AbstractLearner
    approximator::A
end
function (learner::MonteCarloTreeLearner)(env::AbstractEnv)
    actions = action_space(env)
    if ActionStyle(env) == FULL_ACTION_SET
        mask = legal_action_space_mask(env)
        actions = actions[findall(mask)]
    end
    [learner.approximator(state(child(env, a))) for a in actions]
end
# (learner::MonteCarloTreeLearner)(s) = learner.approximator(s)
# (learner::MonteCarloTreeLearner)(s, a) = child()
