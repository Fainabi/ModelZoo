using ReinforcementLearning
using ReinforcementLearning.RLZoo: FirstVisit, EveryVisit, OrdinaryImportanceSampling, WeightedImportanceSampling
using ReinforcementLearning.RLCore: countmap

# _ReinforcementLearning.jl (v0.10.0)_ did not provide implementation of off policy MC with `EVERY_VISIT`.
# and neither MC off-policy Q control.
# Extend off-policy Monte-Carlo methods for EVERY_VISIT and QBasedPolicy
# the code templates are from `ReinforcementLearningZoo: monte_carlo_learner.jl`

"""
    MC Every-Visit Off-Policy V Control
"""


function RLZoo._update!(
    ::EveryVisit,
    ::Tuple{
        <:Union{TabularVApproximator,LinearVApproximator},
        <:Union{TabularVApproximator,LinearVApproximator},
    },
    ::OrdinaryImportanceSampling,
    L::MonteCarloLearner,
    t::AbstractTrajectory,
)
    S, R, W = t[:state], t[:reward], t[:weight]
    (V, G), g, γ, ρ = L.approximator, 0.0, L.γ, 1.0

    for i in length(R):-1:1
        s, r = S[i], R[i]
        g = γ * g + r
        ρ *= W[i]
        update!(G, s => G(s) - ρ * g)
        update!(V, s => V(s) - G(s))
    end
end

function RLZoo._update!(
    ::EveryVisit,
    ::Tuple,
    ::WeightedImportanceSampling,
    L::MonteCarloLearner,
    t::AbstractTrajectory,
)
    S, R, W = t[:state], t[:reward], t[:weight]
    (V, G, Ρ), g, γ, ρ = L.approximator, 0.0, L.γ, 1.0

    for i in length(R):-1:1
        s, r = S[i], R[i]
        g = γ * g + r
        ρ *= W[i]
        update!(G, s => G(s) - ρ * g)
        update!(Ρ, s => Ρ(s) - ρ)
        val = Ρ(s) == 0 ? 0 : G(s) / Ρ(s)
        update!(V, s => V(s) - val)
    end
end

"""
    MC Off-Policy Q Control
"""

function RLBase.update!(::QBasedPolicy{<:MonteCarloLearner}, ::AbstractTrajectory) end

function RLBase.update!(
    p::QBasedPolicy{<:MonteCarloLearner},
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PostEpisodeStage,
)
    update!(p.learner, t)
end

function RLBase.update!(
    t::AbstractTrajectory,
    ::NamedPolicy{<:QBasedPolicy{<:MonteCarloLearner}},
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    empty!(t)
end

function RLBase.update!(
    t::VectorWSARTTrajectory,
    ::OffPolicy{<:QBasedPolicy{<:MonteCarloLearner}},
    env::AbstractEnv,
    s::PreEpisodeStage,
)
    empty!(t)
end

function RLBase.update!(
    t::VectorWSARTTrajectory,
    ::OffPolicy{<:QBasedPolicy{<:MonteCarloLearner}},
    env::AbstractEnv,
    s::PostEpisodeStage,
)
    action = rand(action_space(env))

    push!(t[:state], state(env))
    push!(t[:action], action)
    push!(t[:weight], 1.0)
end



function RLZoo._update!(
    ::FirstVisit,
    ::Tuple{
        <:Union{TabularQApproximator,LinearQApproximator},
        <:Union{TabularQApproximator,LinearQApproximator},
    },
    ::OrdinaryImportanceSampling,
    L::MonteCarloLearner,
    t::AbstractTrajectory,
)
    S, R, W, A = t[:state], t[:reward], t[:weight], t[:action]
    (Q, G), g, γ, ρ = L.approximator, 0.0, L.γ, 1.0
    seen_states = RLCore.countmap((zip(S[1:end-1], A[1:end-1])))

    # @info "debug" S R W A seen_states G Q
    for i in length(R):-1:1
        s, a, r = S[i], A[i], R[i]
        g = γ * g + r
        ρ *= W[i]
        if seen_states[(s, a)] == 1  # first visit
            update!(G, (s, a) => G(s, a) - ρ * g)
            update!(Q, (s, a) => Q(s, a) - G(s, a))
        else
            seen_states[(s, a)] -= 1
        end
    end
end

function RLZoo._update!(
    ::FirstVisit,
    ::Tuple{
        <:Union{TabularQApproximator,LinearQApproximator},
        <:Union{TabularQApproximator,LinearQApproximator},
        <:Union{TabularQApproximator,LinearQApproximator},
    },
    ::WeightedImportanceSampling,
    L::MonteCarloLearner,
    t::AbstractTrajectory,
)
    S, R, W, A = t[:state], t[:reward], t[:weight], t[:action]
    (Q, G, Ρ), g, γ, ρ = L.approximator, 0.0, L.γ, 1.0
    seen_states = countmap((zip(S[1:end-1], A[1:end-1])))

    for i in length(R):-1:1
        s, r, a = S[i], R[i], A[i]
        g = γ * g + r
        ρ *= W[i]
        if seen_states[(s,a)] == 1  # first visit
            update!(G, (s, a) => G(s, a) - ρ * g)
            update!(Ρ, (s, a) => Ρ(s, a) - ρ)
            val = Ρ(s, a) == 0 ? 0 : G(s, a) / Ρ(s, a)
            update!(Q, (s, a) => Q(s, a) - val)
        else
            seen_states[(s,a)] -= 1
        end
    end
end


function RLZoo._update!(
    ::EveryVisit,
    ::Tuple{
        <:Union{TabularQApproximator,LinearQApproximator},
        <:Union{TabularQApproximator,LinearQApproximator},
    },
    ::OrdinaryImportanceSampling,
    L::MonteCarloLearner,
    t::AbstractTrajectory,
)
    S, R, W, A = t[:state], t[:reward], t[:weight], t[:action]
    (Q, G), g, γ, ρ = L.approximator, 0.0, L.γ, 1.0

    for i in length(R):-1:1
        s, r, a = S[i], R[i], A[i]
        g = γ * g + r
        ρ *= W[i]
        update!(G, (s,a) => G(s, a) - ρ * g)
        update!(Q, (s,a) => Q(s, a) - G(s, a))
    end
end

function RLZoo._update!(
    ::EveryVisit,
    ::Tuple{
        <:Union{TabularQApproximator,LinearQApproximator},
        <:Union{TabularQApproximator,LinearQApproximator},
        <:Union{TabularQApproximator,LinearQApproximator},
    },
    ::WeightedImportanceSampling,
    L::MonteCarloLearner,
    t::AbstractTrajectory,
)
    S, R, W, A = t[:state], t[:reward], t[:weight], t[:action]
    (Q, G, Ρ), g, γ, ρ = L.approximator, 0.0, L.γ, 1.0

    for i in length(R):-1:1
        s, a, r = S[i], A[i], R[i]
        g = γ * g + r
        ρ *= W[i]
        update!(G, (s, a) => G(s, a) - ρ * g)
        update!(Ρ, (s, a) => Ρ(s, a) - ρ)
        val = Ρ(s, a) == 0 ? 0 : G(s, a) / Ρ(s, a)
        update!(Q, (s, a) => Q(s, a) - val)
    end
end


# PrioritizedSweepingSamplingModel
#   Simply copy the source codes in `td_learner.jl` but
#   change for loop in updating priority
function RLBase.update!(
    p::QBasedPolicy{<:TDLearner},
    m::PrioritizedSweepingSamplingModel,
    ::AbstractTrajectory,
    env::AbstractEnv,
    ::Union{PreActStage,PostEpisodeStage},
)
    if p.learner.method == :SARS
        transition = sample(m)
        if !isnothing(transition)
            s, a, r, t, s′ = transition
            traj = VectorSARTTrajectory()
            push!(traj; state = s, action = a, reward = r, terminal = t)
            push!(traj; state = s′, action = a)  # here a is a dummy one
            update!(p.learner, traj, env, t ? POST_EPISODE_STAGE : PRE_ACT_STAGE)

            # update priority
            for (s̄, ā, r̄, d̄) in get(m.predecessors, s, [])
                P = RLBase.priority(p.learner, (s̄, ā, r̄, d̄, s))
                if P ≥ m.θ
                    m.PQueue[(s̄, ā)] = P
                end
            end
        end
    else
        @error "unsupported method $(p.learner.method)"
    end
end

