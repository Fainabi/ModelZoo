include("hook.jl")
include("trajectory.jl")

abstract type AbstractSokobanAgent end
function Base.show(io::IO, sokoban_agent::AbstractSokobanAgent)
    show(io, sokoban_agent.game)
    println()
    show(io, sokoban_agent.agent)
end
function Base.run(sokoban_agent::AbstractSokobanAgent, episode=100; hook=OptimalTrajectoryHook())
    run(sokoban_agent.agent, sokoban_agent.game, StopAfterEpisode(episode), hook)
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
    policy = if agent.policy isa OffPolicy
        agent.policy.π_target
    else
        agent.policy
    end
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
        action = policy(game)
        
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
    policy = if agent.policy isa OffPolicy
        agent.policy.π_target
    else
        agent.policy
    end

    approximator = if policy.learner isa DoubleLearner
        # choose one
        policy.learner.L1.approximator
    elseif policy.learner.approximator isa Tuple  # off policy
        policy.learner.approximator[1]
    else
        policy.learner.approximator
    end
    while true
        clear!()
        draw_symbols(game.now_game)

        s = encode_state(game.env_model, game.now_game.player_pos, game.now_game.box_pos)
        # print relate four states' or actions' values
        for a in 1:4
            (r, _, s′), _ = game.env_model(s, a)[1]

            if approximator isa TabularVApproximator
                println(action_name[a], ": ", approximator(s′), ", reward: ", r)
            elseif approximator isa TabularQApproximator
                println(action_name[a], ": ", approximator(s, a), ", reward: ", r)
            end
        end

        # print policy's action
        if policy isa QBasedPolicy
            pr = [prob(policy, game, a) for a in action_space(game)]
            println("π: ", action_name[policy.explorer(pr)])
        else
            println("π: ", action_name[policy(game)])
        end

        # command from terminal
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

        game(action)  # renew state
    end
end

include("policy_iteration.jl")
include("monte-carlo.jl")
include("td.jl")
