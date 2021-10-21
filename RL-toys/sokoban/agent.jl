include("sokoban.jl")
include("utils.jl")
using ReinforcementLearning
using Flux
using ProgressMeter

action_state_map = Dict(
    CartesianIndex(-1, 0) => 1,  # up
    CartesianIndex(0, 1) => 2,  # right
    CartesianIndex(1, 0) => 3,  # down
    CartesianIndex(0, -1) => 4  # left
)
actions = [
    CartesianIndex(-1, 0),
    CartesianIndex(0, 1),
    CartesianIndex(1, 0),
    CartesianIndex(0, -1)
]

action_name = [
    "up", "right", "down", "left"
]

"""
    SokobanEnvModel
"""

mutable struct SokobanEnvModel <: AbstractEnvironmentModel
    state_position
    position_state
    reachable_unit
    reachable_dict

    goal_pos
    end_reward
    step_reward
end

function SokobanEnvModel(skb::Sokoban; end_reward=10., step_reward=0.)
    reachable = skb.reachable_units
    boxes = skb.box_pos
    state_position, position_state = binomial_order(length(reachable)-1, length(boxes))

    SokobanEnvModel(state_position, position_state, skb.reachable_units, skb.reachable_dict, skb.goal_pos, end_reward, step_reward)
end

is_terminated(skb_env::SokobanEnvModel, box_poses) = all(box -> box in skb_env.goal_pos, box_poses)
function Base.show(io::IO, skb_env::SokobanEnvModel)
    println(io, "state space size: ", length(skb_env.position_state))
end

function encode_state(skb_env::SokobanEnvModel, player_pos, boxes_pos)
    player_pos_num = skb_env.reachable_dict[player_pos]
    box_pos_num = sort([skb_env.reachable_dict[pos] for pos in boxes_pos])
    box_pos_num = map(box_pos_num) do x
        if x > player_pos_num
            x - 1
        else
            x
        end
    end
    (player_pos_num - 1)*length(skb_env.state_position) + skb_env.position_state[box_pos_num]
end

function decode_state(skb_env::SokobanEnvModel, state::Int)
    boxes_state = state % length(skb_env.state_position)
    if boxes_state == 0
        boxes_state = length(skb_env.state_position)
    end
    player_state = Int((state - boxes_state) / length(skb_env.state_position)) + 1
    boxes_position = map(skb_env.state_position[boxes_state]) do pos
        if pos < player_state
            pos
        else
            pos + 1
        end
    end
    (skb_env.reachable_unit[player_state], skb_env.reachable_unit[boxes_position])
end

encode_action(move) = action_state_map[move]
decode_action(action) = actions[action]


function (m::SokobanEnvModel)(s::Int, a::Int)
    player_pos, box_poses = decode_state(m, s)
    a = decode_action(a)
    next_pos = player_pos + a
    next_next_pos = next_pos + a


    if next_pos ∉ m.reachable_unit # wall
        # do nothing        
    elseif next_pos in box_poses  # encounter a box
        if next_next_pos ∉ m.reachable_unit || next_next_pos in box_poses  # double boxes or touching wall
            # do nothing
        else  # movable
            player_pos = next_pos
            idx = findfirst(isequal(next_pos), box_poses)
            box_poses[idx] = next_next_pos
        end
    else  # empty unit or goal
        player_pos = next_pos
    end

    r = is_terminated(m, box_poses) ? m.end_reward : m.step_reward  # end game with reward of 1

    # strategy to choose r and iteration, see readme
    [(r, is_terminated(m, box_poses), encode_state(m, player_pos, box_poses)) => 1.0]  # deteministic process
end

RLBase.state_space(m::SokobanEnvModel) = 
    Base.OneTo(length(m.reachable_unit) * length(m.state_position))

RLBase.action_space(_m::SokobanEnvModel) = Base.OneTo(4)



"""
    Agent
"""
mutable struct SokobanAgent
    origin_game::Sokoban

    now_game::Sokoban

    env_model::SokobanEnvModel

    V
    π
end

function new_game(filename)
    game = from_file(filename)
    env_model = SokobanEnvModel(game)
    V = TabularVApproximator(n_state=length(state_space(env_model)), opt=Descent(1.0))
    # π = TabularRandomPolicy(table=Dict(s => fill(0.25, 4) for s in 1:length(state_space(env_model))))
    π = RandomPolicy(action_space(env_model))

    SokobanAgent(copy(game), game, env_model, V, π)
end

reset!(skb_agent::SokobanAgent) = skb_agent.V.table .= 0;
function Base.show(io::IO, skb_agent::SokobanAgent)
    draw_symbols(skb_agent.origin_game)
    show(skb_agent.env_model)
end

function replay(agent::SokobanAgent; time_gap=0.5, max_step=100)
    agent.now_game = copy(agent.origin_game)

    Base.run(`cmd /c cls`)
    draw_symbols(agent.now_game)
    step = 0
    while !is_terminated(agent.now_game) && step < max_step
        step += 1
        sleep(time_gap)
        Base.run(`cmd /c cls`)


        s = encode_state(agent.env_model, agent.now_game.player_pos, agent.now_game.box_pos)
        v = map(action_space(agent.env_model)) do a
            (_, _, s′), _ = agent.env_model(s, a)[1]
            agent.V(s′)
        end

        v, action = findmax(v)  # find optimal policy
        
        step!(agent.now_game, action)

        s′ = encode_state(agent.env_model, agent.now_game.player_pos, agent.now_game.box_pos)
        
        draw_symbols(agent.now_game)
        println("Action: ", action_name[action], ", BeforeState: ", s, ", NowState: ", s′)
        if s == s′  # get stucked
            println("Stucked.")
            break
        end
    end
end


sweep(agent::SokobanAgent, n) = @showprogress 0.5 "Evaluating..." for _ in 1:n 
    policy_evaluation!(V=agent.V, π=agent.π, model=agent.env_model, γ=1.0, θ=Inf64)
end

# one can see how the value changes if we perform value iteration
value_iteration(agent::SokobanAgent; max_iter=10) = @showprogress 0.5 "Value Iterating..." for _ in 1:max_iter
    RLZoo.value_iteration!(V=agent.V, model=agent.env_model, γ=1.0, max_iter=1)
end


function interact!(agent::SokobanAgent) 
    agent.now_game = copy(agent.origin_game)

    while true
        Base.run(`cmd /c cls`)
        draw_symbols(agent.now_game)

        s = encode_state(agent.env_model, agent.now_game.player_pos, agent.now_game.box_pos)
        for a in 1:4
            (_, _, s′), _ = agent.env_model(s, a)[1]
            println(action_name[a], ": ", agent.V(s′))
        end

        line = readline()
        if length(line) == 0
            buf = 'x'
        else
            buf = line[1]
        end

        if buf == 'w'
            up!(agent.now_game)
        elseif buf == 's'
            down!(agent.now_game)
        elseif buf == 'a'
            left!(agent.now_game)
        elseif buf == 'd'
            right!(agent.now_game)
        elseif buf == 'q'
            break
        end
    end
end
