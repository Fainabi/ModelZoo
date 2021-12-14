Base.@kwdef struct RenderHook <: AbstractHook 
    close_at_end::Bool = true  # if it is set to false, one should close the env manually
end

function (h::RenderHook)(::PostActStage, agent, env)
    render(env)
end

function (h::RenderHook)(::PostExperimentStage, agent, env)
    if h.close_at_end
        close(env)
    end
end
