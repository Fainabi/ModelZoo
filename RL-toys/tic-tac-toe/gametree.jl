using ReinforcementLearning
using Flux

"""
    The Monte-Carlo Tree Approximator is a Value Based Approximator
"""

Base.@kwdef struct MonteCarloTreeApproximator{T, O} <: AbstractApproximator
    tree::T
    optimizer::O = Flux.Optimise.InvDecay(1.0)
end

function (app::MonteCarloTreeApproximator)(s)
    node = get(app.tree.id_map, s, nothing)
    if isnothing(node)
        0.0
    else
        reward(node)
    end
end
function RLBase.update!(app::MonteCarloTreeApproximator, correction::Pair{Int,Float64})
    id, e = correction
    x = @view app.tree(id).val[1]
    x̄ = @view [e][1]
    get!(app.optimizer.state, x, 0)
    Flux.Optimise.update!(app.optimizer, x, x̄)
end


Base.@kwdef struct MonteCarloTreeLearner{A} <: AbstractLearner
    approximator::A
end
function (learner::MonteCarloTreeLearner)(env::AbstractEnv; full_return=false)
    actions = action_space(env)

    if full_return
        if ActionStyle(env) == FULL_ACTION_SET
            mask = legal_action_space_mask(env)
            actions = actions[findall(mask)]
        end
        s = [state(child(env, a)) for a in actions]
        v = learner.approximator.(s)
        visited_num = visited(learner.approximator.tree, s)
        v, visited_num, s, actions
    else
        if ActionStyle(env) == FULL_ACTION_SET
            mask = legal_action_space_mask(env)
        else
            mask = ones(Bool, size(actions))
        end
        map(actions, mask) do a, m
            if m
                learner.approximator(state(child(env, a)))
            else
                0.0
            end
        end
    end
end
function RLBase.update!(learner::MonteCarloTreeLearner, correction::Pair{Int,Float64})
    update!(learner.approximator, correction)
end
