
"""
    BatchVectorSARTTrajectory
"""
mutable struct BatchTDVectorTrajectory <: AbstractTrajectory
    batchsize::Int
    traces::Trajectory
    V::TabularVApproximator
    step::Int
    visited::Set

    BatchTDVectorTrajectory(n_state::Int, batchsize::Int) = new(
        batchsize, 
        VectorSARTTrajectory(), 
        TabularVApproximator(n_state=n_state), 
        0, Set())
end

RLCore.@forward BatchTDVectorTrajectory.traces Base.getindex, Base.keys

function RLCore.empty!(t::BatchTDVectorTrajectory)
    t.step = 0
    empty!(t.visited)
    empty!(t.V.optimizer.state)
    t.V.table .= 0
    empty!(t.traces)
end

function RLZoo._update!(
    L::TDLearner,
    ::Union{TabularVApproximator,LinearVApproximator},
    ::Val{:SRS},
    t::BatchTDVectorTrajectory,
    ::PostEpisodeStage,
)
    S = t[:state]
    R = t[:reward]
    n, γ, V = L.n, L.γ, L.approximator
    G = 0.0
    for i in 1:min(n + 1, length(R))
        G = R[end-i+1] + γ * G
        s = S[end-i]
        update!(V, s => V(s) - G)
    end
end

function RLZoo._update!(
    L::TDLearner,
    ::Union{TabularVApproximator,LinearVApproximator},
    ::Val{:SRS},
    t::BatchTDVectorTrajectory,
    ::PreActStage,
)
    S = t[:state]
    R = t[:reward]

    n, γ, V = L.n, L.γ, L.approximator
    if length(R) >= n + 1
        s, s′ = S[end-n-1], S[end]
        G = discount_rewards_reduced(@view(R[end-n:end]), γ) + γ^(n + 1) * V(s′)
        if haskey(t, :weight)
            W = t[:weight]
            @views w = reduce(*, W[end-n-1:end-1])
        else
            w = 1.0
        end

        # batch store
        get!(t.V.optimizer.state, s, 0)  # omit init value in average
        update!(t.V, s => t.V(s) - G)
        
        push!(t.visited, s)
        t.step += 1
    end

    if t.step >= t.batchsize
        t.step = 0
        for s in t.visited
            update!(V, s => w*(V(s) - t.V(s)))
        end
        empty!(t.V.optimizer.state)
        t.V.table .= 0
        empty!(t.visited)
    end
end
