include("sokoban.jl")
include("utils.jl")
using ReinforcementLearning


action_state_map = Dict(
    CartesianIndex(0, 1) => 1,
    CartesianIndex(1, 0) => 2,
    CartesianIndex(0, -1) => 3,
    CartesianIndex(-1, 0) => 4
)
action_space = [
    CartesianIndex(0, 1),
    CartesianIndex(1, 0),
    CartesianIndex(0, -1),
    CartesianIndex(-1, 0)
]

mutable struct SokobanEnvModel <: AbstractEnvironmentModel
    state_position
    position_state
    reachable_unit
    reachable_dict
end

function SokobanEnvModel(skb::Sokoban)
    reachable = skb.reachable_units
    boxes = skb.box_pos
    state_position, position_state = binomial_order(length(reachable)-1, length(boxes))

    SokobanEnvModel(state_position, position_state, skb.reachable_units, skb.reachable_dict)
end


function encode_state(skb_env::SokobanEnvModel, player_pos, boxes_pos)
    (skb_env.reachable_dict[player_pos] - 1)*length(skb_env.state_position) + 
        skb_env.position_state[[skb_env.reachable_dict[pos] for pos in boxes_pos]]
end

function decode_state(skb_env::SokobanEnvModel, state::Int)
    boxes_state = state % length(skb_env.state_position)
    if boxes_state == 0
        boxes_state = length(skb_env.state_position)
    end
    player_state = Int((state - boxes_state) / length(skb_env.state_position)) + 1
    boxes_state = map(skb_env.state_position[boxes_state]) do pos
        if pos < player_state
            pos
        else
            pos + 1
        end
    end
    (skb_env.reachable_unit[player_state], skb_env.reachable_unit[skb_env.state_position[boxes_state]...])
end

encode_action(move) = action_state_map[move]
decode_action(action) = action_space[action]


function (m::SokobanEnvModel)(s::Int, a::Int)

end

RLBase.state_space(m::SokobanEnvModel) = 
    Base.OneTo(length(m.reachable_units) * binomial(length(m.reachable_units)-1, m.boxes))

RLBase.action_space(m::SokobanEnvModel) = Base.OneTo(4)