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
    function TD(game::SokobanGame, method; α=0.5, γ=1.0, ϵ=0.1)
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
                # n=0  # TD(0)
            ),
            explorer=EpsilonGreedyExplorer(ϵ)
        )

        new(game, Agent(policy=policy, trajectory=VectorSARTTrajectory()))
    end
end

SARSA(game::SokobanGame; α=0.5, γ=1.0, ϵ=0.1) = TD(game, :SARSA; α=α, γ=γ, ϵ=ϵ)
SARS(game::SokobanGame; α=0.5, γ=1.0, ϵ=0.1) = TD(game, :SARS; α=α, γ=γ, ϵ=ϵ)
ExpectedSARSA(game::SokobanGame; α=0.5, γ=1.0, ϵ=0.1) = TD(game, :ExpectedSARSA; α=α, γ=γ, ϵ=ϵ)


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



# TD(0) is a prediction method
mutable struct SRS <: AbstractSokobanAgent
    game::SokobanGame
    agent::Agent
    function SRS(game; α=0.5, γ=1.0, ϵ=0.1)
        n_state = length(state_space(game))

        V = TabularVApproximator(
            n_state=n_state,
            opt=Descent(α)
        )

        policy = VBasedPolicy(
            learner=TDLearner(
                approximator=V,
                method=:SRS,
                γ=γ
            ),
        )

        new(game, Agent(policy=policy, trajectory=VectorSARTTrajectory()))
    end
end

