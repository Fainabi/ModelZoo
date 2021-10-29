using ReinforcementLearning
using Flux.Optimise: InvDecay

"""
    Monte Carlo methods for learning V and Q functions.

Including: α-MC methods, on policy and off policy.
The off policy contains:
    - ordinary importance sampling,
    - weighted importance sampling,
    - discounting-aware importance sampling,
    - per-decision importance sampling
"""
function MC(game::SokobanGame, type=:V; sampling=NO_SAMPLING, kind=FIRST_VISIT, γ=1.0, ϵ=0.05)
    n_state = length(state_space(game))
    n_action = length(action_space(game))
    explorer = EpsilonGreedyExplorer(ϵ)

    if sampling == NO_SAMPLING  # on policy
        if type == :V
            V = TabularVApproximator(
                n_state=n_state,
                init=0.0,
                opt=InvDecay(1.0),  # average visited returns
            )

            VBasedPolicy(
                learner = MonteCarloLearner(
                    approximator=V, 
                    sampling=sampling,
                    γ=γ, 
                    kind=kind,
                ),
                mapping = function (env, V)
                    A = legal_action_space(env)
                    values = map(a -> V(child(env, a)), A)
                    A[explorer(values)]
                end
            )
        else
            Q = TabularQApproximator(
                n_state=n_state,
                n_action=n_action,
                init=0.0,
                opt=InvDecay(1.0)
            )

            QBasedPolicy(
                learner = MonteCarloLearner(
                    approximator=Q,
                    sampling=sampling,
                    γ=γ,
                    kind=kind
                ),
                explorer = explorer
            )
        end

    else  # off policy
        if type == :V

        end  # ReinforcementLearning.jl did not implement EVERY_VISIT for off-policy learning
    end
end