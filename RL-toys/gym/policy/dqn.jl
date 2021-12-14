using Flux

function DeepQLearningAgent(n_state, n_action; ϵ=0.01, capacity=1000)
    model = Chain(
        Dense(n_state, 10, relu),
        Dense(10, 10, relu),
        Dense(10, n_action)
    )
    app = NeuralNetworkApproximator(model = model, optimizer = ADAM())

    learner = BasicDQNLearner(
        approximator=app,
        loss_func=Flux.Losses.huber_loss
    )
    policy = QBasedPolicy(
        learner = learner,
        explorer = EpsilonGreedyExplorer(ϵ)
    )

    Agent(
        policy=policy, 
        trajectory=CircularArraySARTTrajectory(state=Vector{Float32} => (n_state,), capacity=capacity)
    )
end

function DeepQLearningAgent(env::AbstractEnv; kwargs...)
    DeepQLearningAgent(length(state(env)), length(action_space(env)); kwargs...)
end

