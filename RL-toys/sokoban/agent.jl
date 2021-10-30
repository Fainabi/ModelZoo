abstract type AbstractSokobanAgent end


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
        for t in agent.trajectory[:terminal]
            push!(h.trajectory[:terminal], t)
        end
        for w in agent.trajectory[:weight]
            push!(h.trajectory[:weight], w)
        end
        for r in agent.trajectory[:reward]
            push!(h.trajectory[:reward], r)
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

function replay(game::SokobanGame, trajectory=Trajectory; time_gap=0.5)
    game.now_game = copy(game.origin_game)

    clear!()
    
    s = trajectory[:state]
    a = trajectory[:action]
    # r = trajectory[:reward]
    # t = trajectory[:terminal]
    if isempty(s)
        return
    end
    load_state(game, s[1])
    draw_symbols(game.now_game)
    println("Action: ", action_name[a[1]])
    quit = false

    task = Base.Threads.@spawn (() -> begin
        while !quit
            c = get_keypress()
            if c == 'q' || c == 'Q'
                quit = true
                break
            end
        end
    end)()

    for idx in 2:length(s)
        if quit
            break
        end
        sleep(time_gap)
        clear!()
        load_state(game, s[idx])
        draw_symbols(game.now_game)
        println("Action: ", action_name[a[idx-1]])
    end
    if !quit
        schedule(task, InterruptException(), error=true);
    end
    nothing
end

function replay(skb_agent::AbstractSokobanAgent, hook::OptimalTrajectoryHook; time_gap=0.5)
    replay(skb_agent.game, hook.trajectory; time_gap=time_gap)
end

function replay(skb_agent::AbstractSokobanAgent; time_gap=0.5)
    game = skb_agent.game
    RLBase.reset!(game)
    agent = skb_agent.agent
    max_step = game.max_steps

    quit = false
    task = Base.Threads.@spawn (() -> begin
        while !quit
            c = get_keypress()
            if c == 'q' || c == 'Q'
                quit = true
                break
            end
        end
    end)()

    clear!()
    draw_symbols(game.now_game)
    step = 0
    while !is_terminated(game.now_game) && step < max_step
        if quit
            break
        end
        step += 1
        sleep(time_gap)
        clear!()

        s = game.state
        action = agent(game)
        
        step!(game.now_game, action)
        game(action)
        s′ = game.state
        
        draw_symbols(game.now_game)
        println("Action: ", action_name[action], ", BeforeState: ", s, ", NowState: ", s′)
        # it may get stucked, but mc can explore it
    end

    if !quit
        schedule(task, InterruptException(), error=true);
    end
    nothing
end

function interact!(skb_agent::AbstractSokobanAgent)
    game = skb_agent.game
    RLBase.reset!(game)
    agent = skb_agent.agent

    while true
        clear!()
        draw_symbols(game.now_game)

        s = encode_state(game.env_model, game.now_game.player_pos, game.now_game.box_pos)
        for a in 1:4
            (r, _, s′), _ = game.env_model(s, a)[1]
            if agent.policy.learner.approximator isa TabularVApproximator
                println(action_name[a], ": ", agent.policy.learner(s′), ", reward: ", r)
            else
                # TabularQApproximator
            end
        end
        println("π: ", action_name[agent(game)])

        line = readline()
        if length(line) == 0
            buf = 'x'
        else
            buf = line[1]
        end

        action = if buf == 'w'
            up!(game.now_game)
            action_state_map[UP]
        elseif buf == 's'
            down!(game.now_game)
            action_state_map[DOWN]
        elseif buf == 'a'
            left!(game.now_game)
            action_state_map[LEFT]
        elseif buf == 'd'
            right!(game.now_game)
            action_state_map[RIGHT]
        elseif buf == 'q'
            break
        else
            up!(game.now_game)
            action_state_map[UP]
        end

        game(action)
    end
end

include("policy_iteration.jl")
include("monte-carlo.jl")
