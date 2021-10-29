using ReinforcementLearning: TabularVApproximator
using Flux.Optimise: InvDecay
using ProgressMeter: @showprogress

"""
    Dynamic Programming for Sokoban
"""
mutable struct DP 
    game::SokobanGame

    V::TabularApproximator
    π::AbstractPolicy
    function DP(game::SokobanGame, policy_type=:RandomPolicy)
        π = RandomPolicy(action_space(game.env_model))
        if policy_type == :TabularPolicy
            π = TabularPolicy(
                table=Dict(zip(state_space(game.env_model), fill(1, length(state_space(game.env_model))))), 
                n_action=length(action_space(game.env_model)))
        end
        # in dp, we mostly care about V functions
        V = TabularVApproximator(n_state=length(state_space(game.env_model)), opt=InvDecay(1.0))
        new(game, V, π)
    end
end

function Base.show(io::IO, dp::DP)
    show(io, dp.game)
    println()
    show(io, dp.V)
    println()
    println()
    show(io, dp.π)
end

function reset!(dp::DP)
    reset!(dp.game)
    fill!(dp.V.table, 0.)
    if dp.π isa TabularPolicy
        dp.π = TabularPolicy(
            table=Dict(zip(state_space(game.env_model), fill(1, length(state_space(game.env_model))))), 
            n_action=length(action_space(game.env_model)))
    end
end


sweep(agent::DP, n) = @showprogress 0.5 "Evaluating..." for _ in 1:n 
    policy_evaluation!(V=agent.V, π=agent.π, model=agent.game.env_model, γ=1.0, θ=Inf64)
end

policy_iterate(agent::DP, n) = @showprogress 0.5 "Policy Iterating..." for _ in 1:n
    policy_evaluation!(V=agent.V, π=agent.π, model=agent.game.env_model, γ=1.0, θ=Inf64)
    policy_improvement!(V=agent.V, π=agent.π, model=agent.game.env_model, γ=1.0)
end

# one can see how the value changes if we perform value iteration
value_iteration(agent::DP; max_iter=10) = @showprogress 0.5 "Value Iterating..." for _ in 1:max_iter
    RLZoo.value_iteration!(V=agent.V, model=agent.game.env_model, γ=1.0, max_iter=1)
end


function replay(agent::DP; time_gap=0.5, max_step=100)
    game = agent.game
    game.now_game = copy(game.origin_game)

    clear!()
    draw_symbols(game.now_game)
    step = 0
    while !is_terminated(game.now_game) && step < max_step
        step += 1
        sleep(time_gap)
        clear!()


        s = encode_state(game.env_model, game.now_game.player_pos, game.now_game.box_pos)
        if agent.π isa TabularPolicy
            action = agent.π(s)
        else
            v = map(action_space(game.env_model)) do a
                (_, _, s′), _ = game.env_model(s, a)[1]
                agent.V(s′)
            end
    
            v, action = findmax(v)  # find optimal policy
        end
        
        step!(game.now_game, action)

        s′ = encode_state(game.env_model, game.now_game.player_pos, game.now_game.box_pos)
        
        draw_symbols(game.now_game)
        println("Action: ", action_name[action], ", BeforeState: ", s, ", NowState: ", s′)
        if s == s′  # get stucked
            println("Stucked.")
            break
        end
    end
end


function interact!(dp::DP) 
    game = dp.game
    game.now_game = copy(game.origin_game)

    while true
        clear!()
        draw_symbols(game.now_game)

        s = encode_state(game.env_model, game.now_game.player_pos, game.now_game.box_pos)
        for a in 1:4
            (_, _, s′), _ = game.env_model(s, a)[1]
            println(action_name[a], ": ", dp.V(s′))
        end

        line = readline()
        if length(line) == 0
            buf = 'x'
        else
            buf = line[1]
        end

        if buf == 'w'
            up!(game.now_game)
        elseif buf == 's'
            down!(game.now_game)
        elseif buf == 'a'
            left!(game.now_game)
        elseif buf == 'd'
            right!(game.now_game)
        elseif buf == 'q'
            break
        end
    end
end
