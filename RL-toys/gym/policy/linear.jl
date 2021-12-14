function LinearLearnerAgent(n_state, n_action; γ=1.0, n=0, ϵ=0.01, λ=1.0)
    # using Q-Learning
    app = LinearQApproximator(n_state=n_state, n_action=n_action, opt=Descent(λ))
    learner = TDLearner(
        approximator=app,
        method=:SARS,
        γ=γ,
        n=n,
    )
    policy = QBasedPolicy(learner = learner, explorer = EpsilonGreedyExplorer(ϵ))

    Agent(policy=policy, trajectory=VectorSARTTrajectory(state=Any))
end


function LinearLearnerAgent(env::AbstractEnv; kwargs...)
    LinearLearnerAgent(length(state(env)), length(action_space(env)); kwargs...)
end
