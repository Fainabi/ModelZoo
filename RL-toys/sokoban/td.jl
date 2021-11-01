using ReinforcementLearning
using Flux.Optimise: InvDecay, Descent

"""
    Temporal-Difference Learning

Containing:
    - SARSA
    - Q-Learning
    - Expected SARSA
    - Double Learning

Batch updating is implemented.
"""

mutable struct TD <: AbstractSokobanAgent
    game::SokobanGame
    agent::Agent
    function TD(game::SokobanGame, method; α=0.5, γ=1.0, ϵ=0.1, n=0)
        n_state = length(state_space(game))
        n_action = length(action_space(game))

        # control problem, thus QBasedPolicy
        Q = TabularQApproximator(
            n_state=n_state,
            n_action=n_action,
            opt=Descent(α)
        )

        policy = QBasedPolicy(
            learner=TDLearner(
                approximator=Q,
                method=method,
                γ=γ,
                n=n
            ),
            explorer=EpsilonGreedyExplorer(ϵ)
        )

        new(game, Agent(policy=policy, trajectory=VectorSARTTrajectory()))
    end
end

SARSA(game::SokobanGame; α=0.5, γ=1.0, ϵ=0.1, n=0) = TD(game, :SARSA; α=α, γ=γ, ϵ=ϵ, n=n)
SARS(game::SokobanGame; α=0.5, γ=1.0, ϵ=0.1, n=0) = TD(game, :SARS; α=α, γ=γ, ϵ=ϵ, n=n)
ExpectedSARSA(game::SokobanGame; α=0.5, γ=1.0, ϵ=0.1, n=0) = TD(game, :ExpectedSARSA; α=α, γ=γ, ϵ=ϵ, n=n)


mutable struct DoubleSokobanLearner <: AbstractSokobanAgent
    game::SokobanGame
    agent::Agent

    function DoubleSokobanLearner(game::SokobanGame, method1, method2; α=0.5, γ=1.0, ϵ=0.1)
        td1 = TD(game, method1; α=α, γ=γ, ϵ=ϵ)
        td2 = TD(game, method2; α=α, γ=γ, ϵ=ϵ)
        policy = QBasedPolicy(
            learner = DoubleLearner(
                L1=td1.agent.policy.learner,
                L2=td2.agent.policy.learner,
            ),
            explorer = EpsilonGreedyExplorer(ϵ)
        )

        new(game, Agent(policy=policy, trajectory=VectorSARTTrajectory()))
    end
end





# TD(n) is a predictive method
# Trajectory can use BatchTDVectorTrajectory
mutable struct SRS <: AbstractSokobanAgent
    game::SokobanGame
    agent::Agent
    function SRS(game; α=0.5, γ=1.0, n=0, trajectory=VectorSARTTrajectory())
        n_state = length(state_space(game))

        V = TabularVApproximator(
            n_state=n_state,
            opt=Descent(α)
        )

        policy = VBasedPolicy(
            learner=TDLearner(
                approximator=V,
                method=:SRS,
                γ=γ,
                n=n
            ),
        )

        new(game, Agent(policy=policy, trajectory=trajectory))
    end
end

# DynaAgent
mutable struct Dyna <: AbstractSokobanAgent
    game::SokobanGame
    agent::DynaAgent
    function Dyna(game::SokobanGame, method=:SARS; model=ExperienceBasedSamplingModel(), α=0.5, γ=1.0, ϵ=0.1, n=0, plan_step=10)
        n_state = length(state_space(game))
        n_action = length(action_space(game))

        Q = TabularQApproximator(
            n_state=n_state,
            n_action=n_action,
            opt=Descent(α)
        )

        policy = QBasedPolicy(
            learner=TDLearner(
                approximator=Q,
                method=method,
                γ=γ,
                n=n
            ),
            explorer=EpsilonGreedyExplorer(ϵ; is_break_tie=true)
        )

        new(
            game,
            DynaAgent(
                policy=policy,                          # direct RL and acting
                model=model,                            # model learning, Dyna-Q+ uses TimeBasedSamplingModel(;n_actions=4)
                                                        # also consider PrioritizedSweepingSamplingModel()
                trajectory=VectorSARTTrajectory(),
                plan_step=plan_step,                    # for indirect RL or Q planning
            ),
        )
    end
end

