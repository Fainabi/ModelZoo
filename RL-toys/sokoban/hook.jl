mutable struct StatesVisitHook <: AbstractHook
    visit::Dict{Int, Int}
    StatesVisitHook() = new(Dict())
end
function (h::StatesVisitHook)(::PostActStage, agent, env)
    h.visit[state(env)] = get(h.visit, state(env), 0) + 1
end

mutable struct OptimalTrajectoryHook <: AbstractHook
    trajectory::Trajectory
    OptimalTrajectoryHook() = new(VectorWSARTTrajectory())
end
function (h::OptimalTrajectoryHook)(::PostEpisodeStage, agent, env)
    ret, step = trajectory_return(h.trajectory)
    if step == 0
        ret = -Inf64
    end
    new_ret, new_step = trajectory_return(agent.trajectory)
    if ret > new_ret
        return
    elseif ret == new_ret && step < new_step
        return
    else
        empty!(h.trajectory)
        for s in agent.trajectory[:state]
            push!(h.trajectory[:state], s)
        end
        for a in agent.trajectory[:action]
            push!(h.trajectory[:action], a)
        end
        for r in agent.trajectory[:reward]
            push!(h.trajectory[:reward], r)
        end
        if haskey(agent.trajectory, :terminal)
            for t in agent.trajectory[:terminal]
                push!(h.trajectory[:terminal], t)
            end
        end
        if haskey(agent.trajectory, :weight)
            for w in agent.trajectory[:weight]
                push!(h.trajectory[:weight], w)
            end
        end
    end
end

function trajectory_return(trajectory::Trajectory, γ=1.0)
    r = 0
    len = length(trajectory[:reward])
    for v in reverse(trajectory[:reward])
        r = v + γ*r
    end
    r, len
end