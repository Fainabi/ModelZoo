using ReinforcementLearning
using Flux.Optimise: InvDecay

mutable struct MC <: AbstractSokobanAgent
    game::SokobanGame
    agent::Agent

    """
        Monte Carlo methods for learning V and Q functions.

    Including: α-MC methods, on policy and off policy.
    The off policy contains:
        - ordinary importance sampling,
        - weighted importance sampling,
        - discounting-aware importance sampling,
        - per-decision importance sampling
    """
    function MC(game::SokobanGame, type=:V; sampling=NO_SAMPLING, kind=FIRST_VISIT, γ=1.0, ϵ=0.1)
        n_state = length(state_space(game))
        n_action = length(action_space(game))
        explorer = EpsilonGreedyExplorer(ϵ)

        policy = if sampling == NO_SAMPLING  # on policy
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

        # VectorWSARTTrajectory records importance sampling ratio.
        # WSART representing: weight, state, action, reward, ternimated
        # In on policy learning, weight would not be updated.
        new(game, Agent(policy=policy, trajectory=VectorWSARTTrajectory()))
    end
end

function reset!(mc::MC)
    RLBase.reset!(mc.game)
    mc.agent.policy.learner.approximator.table .= 0.0
    empty!(mc.agent.policy.learner.approximator.optimizer.state)
end

function Base.show(io::IO, mc::MC)
    show(mc.game)
    println()
    show(mc.agent)
end

function Base.run(mc::MC, episode=100; hook=OptimalTrajectoryHook())
    run(mc.agent, mc.game, StopAfterEpisode(episode), hook)
end

sweep(mc::MC, n) = @showprogress 0.5 "Evaluating..." for _ in 1:n 
    RLZoo.value_iteration!(
        V=mc.agent.policy.learner.approximator, 
        model=mc.game.env_model, 
        γ=1.0, 
        max_iter=1
    )
end
