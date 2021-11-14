using ReinforcementLearning

Base.@kwdef mutable struct MonteCarloTreeSearchPolicy{L,E,P} <: AbstractPolicy
    learner::L
    explorer::E
    episodes_per_search::Int = 100

    selection::Union{Int, Nothing} = nothing  # The node selected, if all leaves are explorered, choose nothing
    selected_path::Vector{Int} = Int[]
    simulation_policy::AbstractPolicy = RandomPolicy()
    simulation_hook::AbstractHook = RewardsPerEpisode()
    backup_reward_mapping = (G, r) -> G + r
    oppenent_policy::P = RandomPolicy()
end

function (π::MonteCarloTreeSearchPolicy)(env; search_mode=true)
    empty!(π.learner.approximator.tree)
    spawn!(π.learner.approximator.tree, state(env))
    empty!(π.simulation_hook.rewards)

    if search_mode
        # @info "Searching..."
        for epoch in 1:π.episodes_per_search
            # @info "Searching $epoch"
            empty!(π.selected_path)
            π.selection = nothing
            # empty!(π.simulation_hook.rewards)

            newenv = select!(π, env)
            if isnothing(π.selection)
                break
            end
            expand!(π, newenv)
            simulate!(π, newenv)
            backup!(π, newenv)
        end
    end

    π(env, ActionStyle(env), action_space(env))
end
(π::MonteCarloTreeSearchPolicy)(env, ::MinimalActionSet, ::Base.OneTo) = π.explorer(π.learner(env))
(π::MonteCarloTreeSearchPolicy)(env, ::FullActionSet, ::Base.OneTo) = 
    π.explorer(π.learner(env), legal_action_space_mask(env))

# search algorithm does not need to approxiate the values
function RLBase.update!(::MonteCarloTreeSearchPolicy, ::AbstractTrajectory) end


# implementation

function select!(p::MonteCarloTreeSearchPolicy, env::AbstractEnv)
    isleaf = false
    newenv = copy(env)

    while !isleaf && !is_terminated(newenv)
        # track
        push!(p.selected_path, state(newenv))

        v, visited_number, s, actions = p.learner(newenv; full_return=true)
        parent_visited = visited(p.learner.approximator.tree, state(newenv))

        # for starting, randomly choose one
        if parent_visited == 0
            # @info "Parent Not visited"
            i = rand(Base.OneTo(length(s)))
            selection, action = s[i], actions[i]
            p.selection = selection
            newenv(action)
            is_terminated(newenv) || newenv |> p.oppenent_policy |> newenv
            break
        end

        # find the unvisited nodes
        # prevent log(1) / 0
        unvisited = findall(iszero, visited_number)
        if !isempty(unvisited)
            # @info "Some nodes unvisited"
            i = rand(unvisited)
            id, action = s[i], actions[i]
            p.selection = id
            newenv(action)
            is_terminated(newenv) || newenv |> p.oppenent_policy |> newenv
            break
        end

        # compute 
        # @info "Compute UCB"
        ucb = v .+ 2*sqrt.(log(parent_visited) ./ visited_number)
        _, i = findmax(ucb)  # may be inf
        id = s[i]

        isleaf = !isnode(p.learner.approximator.tree, id)
        newenv(actions[i])
        if isleaf
            p.selection = id
        else
            # assert now state(newenv) == s
            push!(p.selected_path, state(newenv))
        end
        is_terminated(newenv) || newenv |> p.oppenent_policy |> newenv
    end


    newenv
end

function expand!(p::MonteCarloTreeSearchPolicy, newenv::AbstractEnv)
    # new state selected from env
    push!(p.selected_path, p.selection)
    for i in 2:length(p.selected_path)
        spawn!(p.learner.approximator.tree, p.selected_path[i-1], p.selected_path[i])
    end
end

function simulate!(p::MonteCarloTreeSearchPolicy, newenv::AbstractEnv)
    run(p.simulation_policy, newenv, StopAfterEpisode(1), p.simulation_hook, true)
end

function backup!(p::MonteCarloTreeSearchPolicy, newenv::AbstractEnv)
    R = p.simulation_hook.rewards[end][end]
    if isodd(length(p.simulation_hook.rewards[end]))
        R = -R
    end
    if R < 0
        R = 0.0
    end
    # @info "Reward" R
    for s in p.selected_path
        visit!(p.learner.approximator.tree, s)
        update!(p.learner, s => p.learner.approximator(s) - R)
    end
end

