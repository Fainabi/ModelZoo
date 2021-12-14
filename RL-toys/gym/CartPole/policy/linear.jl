# the naive model performs normal
function LinearLearnerAgent()
    dim = 4  # dimension of observation

    # using Q-Learning
    app = LinearQApproximator(n_state=dim, n_action=2, opt=Descent(0.5))
    learner = TDLearner(
        approximator=app,
        method=:SARS,
        γ=0.5
    )
    policy = QBasedPolicy(learner = learner, explorer = EpsilonGreedyExplorer(0.01))

    Agent(policy=policy, trajectory=VectorSARTTrajectory(state=Any))
end

