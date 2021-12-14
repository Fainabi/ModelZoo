using Flux

function DeepQLearningAgent()
    model = Chain(
        Dense(4, 10, relu),
        Dense(10, 10, relu),
        Dense(10, 2)
    )
    app = NeuralNetworkApproximator(model = model, optimizer = ADAM())

    learner = BasicDQNLearner(
        approximator=app,
        loss_func=Flux.Losses.huber_loss
    )
    policy = QBasedPolicy(
        learner = learner,
        explorer = EpsilonGreedyExplorer(0.01)
    )

    Agent(policy=policy, trajectory=CircularArraySARTTrajectory(state=Vector{Float32} => (4,), capacity=1000))
end
