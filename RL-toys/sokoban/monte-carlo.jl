using ReinforcementLearning
using Flux.Optimise: InvDecay, Descent

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
            π_t = if type == :V
                V = TabularVApproximator(
                    n_state=n_state,
                    init=0.0,
                    opt=Descent(1.0),  # V is tracked to G
                )
                G = TabularVApproximator(
                    n_state=n_state,
                    init=0.0,
                    opt=InvDecay(1.0),  # returns are averaged
                )


                approximator = if sampling == ORDINARY_IMPORTANCE_SAMPLING
                    (V, G)
                elseif sampling == WEIGHTED_IMPORTANCE_SAMPLING
                    # returns are weighted averaged with P: G / P
                    # see `ReinforcementLearningZoo.monte_carlo_learner.jl`
                    # Thus (∑G)/(∑P) = mean(G)/mean(P)
                    P = TabularVApproximator(
                        n_state=n_state,
                        init=0.0,
                        opt=InvDecay(1.0),
                    )

                    (V, G, P)
                end
                
                VBasedPolicy(
                    learner = MonteCarloLearner(
                        approximator=approximator,
                        kind=kind,
                        sampling=sampling,
                        γ=γ
                    ),
                    mapping = function (env, L)
                        A = legal_action_space(env)
                        values = map(a -> L.approximator[1](state(child(env, a))), A)
                        _, a = findmax(values)
                        A[a]
                    end
                )
            else  # Q functions
                Q = TabularQApproximator(
                    n_state=n_state,
                    n_action=n_action,
                    init=0.0,
                    opt=Descent(1.0),
                )
                G = TabularQApproximator(
                    n_state=n_state,
                    n_action=n_action,
                    init=0.0,
                    opt=InvDecay(1.0),
                )

                approximator = if sampling == ORDINARY_IMPORTANCE_SAMPLING
                    (Q, G)
                elseif sampling == WEIGHTED_IMPORTANCE_SAMPLING
                    P = TabularQApproximator(
                        n_state=n_state,
                        n_action=n_action,
                        init=0.0,
                        opt=InvDecay(1.0)
                    )

                    (Q, G, P)
                end
                
                QBasedPolicy(
                    learner = MonteCarloLearner(
                        approximator=approximator,
                        kind=kind,
                        sampling=sampling,
                        γ=γ,
                    ),
                    explorer = GreedyExplorer()
                )
            end  

            # no polluting the general prob(AbstractPolicy, s, a)
            if type == :V
                @eval function RLBase.prob(π::typeof($π_t), env, a)
                    # greedy
                    π.mapping(env, π.learner) == a
                end 
            elseif type == :Q
                @eval function RLBase.prob(π::typeof($π_t), env::AbstractEnv, a)
                    values = map(a -> π.learner.approximator[1](state(env), a), legal_action_space(env))
                    π.explorer(values) == a
                end

                # for mapping env to action, since a policy with multiple approximators
                # does not have such mapping.
                @eval function (π::typeof($π_t))(env::AbstractEnv)
                    values = map(a -> π.learner.approximator[1](state(env), a), legal_action_space(env))
                    π.explorer(values)
                end
            end

            OffPolicy(
                π_t,
                RandomPolicy(action_space(game))
            )
        end

        # VectorWSARTTrajectory records importance sampling ratio.
        # WSART representing: weight, state, action, reward, ternimated
        # In on policy learning, weight would not be updated.
        new(game, Agent(policy=policy, trajectory=VectorWSARTTrajectory()))
    end
end

function reset_approximator(approximator)
    approximator.table .= 0.0
    if approximator.optimizer isa InvDecay
        empty!(approximator.optimizer.state)
    end
end

function reset!(mc::MC)
    RLBase.reset!(mc.game)
    approximator = mc.agent.policy.learner.approximator
    if approximator isa Tuple
        reset_approximator.(approximator)
    else
        reset_approximator(approximator)
    end
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
    if mc.agent.policy isa VBasedPolicy
        RLZoo.value_iteration!(
            V=mc.agent.policy.learner.approximator, 
            model=mc.game.env_model, 
            γ=1.0, 
            max_iter=1
        )
    elseif mc.agent.policy isa QBasedPolicy
        # value iteration for Q functions
        model = mc.game.env_model
        Q = mc.agent.policy.learner.approximator
        γ = mc.agent.policy.learner.γ
        for s in state_space(model), a in action_space(model)
            q = sum(
                p * (r + (1 - t) * γ * maximum(
                    Q(s′, a′) for a′ in action_space(model)
                )) for ((r, t, s′), p) in model(s, a)
            )
            δ = Q(s, a) - q
            update!(Q, (s,a) => δ)
        end
    end
end
