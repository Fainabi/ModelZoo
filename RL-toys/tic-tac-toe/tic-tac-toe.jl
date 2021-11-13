include("tree.jl")
include("gametree.jl")
include("MCTS.jl")

using ReinforcementLearning
using Flux

env = TicTacToeEnv()

agent = Agent(
    policy = MonteCarloTreeSearchPolicy(
        learner=MonteCarloTreeLearner(
            approximator=MonteCarloTreeApproximator(
                tree=MonteCarloTree{Float64}(),
                optimizer=InvDecay(1.0),
            )
        ),
        explorer=EpsilonGreedyExplorer(0.1),
        episodes_per_search = 100,
    ),
    trajectory = VectorSARTTrajectory(),
)
