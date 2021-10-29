using ReinforcementLearning

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

is_terminated_env(skb_env::SokobanEnvModel, box_poses) = all(box -> box in skb_env.goal_pos, box_poses)
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

    r = is_terminated_env(m, box_poses) ? m.end_reward : m.step_reward  # end game with reward of 1

    # strategy to choose r and iteration, see readme
    [(r, is_terminated_env(m, box_poses), encode_state(m, player_pos, box_poses)) => 1.0]  # deteministic process
end

RLBase.state_space(m::SokobanEnvModel) = 
    Base.OneTo(length(m.reachable_unit) * length(m.state_position))

RLBase.action_space(_m::SokobanEnvModel) = Base.OneTo(4)



"""
    Game Env Wrapping
"""
mutable struct SokobanGame <: AbstractEnv
    origin_game::Sokoban

    now_game::Sokoban

    env_model::SokobanEnvModel

    reward
    state
    terminated
end

#=
    Minimal Configs
=#
RLBase.state_space(skb_game::SokobanGame) = state_space(skb_game.env_model)
RLBase.action_space(skb_game::SokobanGame) = action_space(skb_game.env_model)
RLBase.state(skb_game::SokobanGame) = skb_game.state
RLBase.is_terminated(skb_game::SokobanGame) = skb_game.terminated
RLBase.reward(skb_game::SokobanGame) = skb_game.reward

function (env::SokobanGame)(a::Int)
    s = env.state
    (r, t, s′), _ = env.env_model(s, a)[1]
    env.reward = r
    env.state = s′
    env.terminated = t
end
function RLBase.reset!(game::SokobanGame) 
    game.now_game = copy(game.origin_game)
    game.reward = 0
end


function new_game(filename)
    game = from_file(filename)
    env_model = SokobanEnvModel(game)
    state = encode_state(env_model, game.player_pos, game.box_pos)
    SokobanGame(copy(game), game, env_model, 0, state, false)
end

function Base.show(io::IO, game::SokobanGame)
    draw_symbols(game.origin_game)
    show(game.env_model)
end

function interact!(game::SokobanGame) 
    game.now_game = copy(game.origin_game)

    while true
        clear!()
        draw_symbols(game.now_game)

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