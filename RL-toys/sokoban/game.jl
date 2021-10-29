mutable struct Sokoban
    Map

    box_pos
    goal_pos
    player_pos

    reachable_units
    reachable_dict
end


@enum UnitType begin
    Empty = 0
    Wall = 1
    Goal = 2
    Box = 3
    Player = 4
    BoxOnGoal = 5
    PlayerOnGoal = 6
end


const empty_unit = ' '
const wall_unit = 'X'
const box_unit = 'O'
const goal_unit = '*'
const player_unit = 'P'
const box_goal = '⦿'
const player_goal = '@'

type_to_symbol = Dict(
    Empty => ' ',
    Wall => 'X',
    Box => 'O',
    Goal => '*',
    Player => 'P',
    BoxOnGoal => '⦿',
    PlayerOnGoal => '@'
)

symbol_to_type = Dict(
    ' ' => Empty,
    'X' => Wall,
    'O' => Box,
    '*' => Goal,
    'P' => Player,
    '⦿' => BoxOnGoal,
    '@' => PlayerOnGoal
)

const UP = CartesianIndex(-1, 0)
const RIGHT = CartesianIndex(0, 1)
const DOWN = CartesianIndex(1, 0)
const LEFT = CartesianIndex(0, -1)



"""
    Sokoban Map needs to be a matrix containing types of each unit: wall, box or goal.
Unit outside of the walls are also wall.
"""
function Sokoban(sokoban_map)
    # Since it is kind of hard to direct decode integer to state matrix, 
    # we first encode each state, and store them with unique code number

    box_init_pos = []
    goal_pos = []
    player_pos = []
    reachable_units = []
    reachable_dict = Dict()

    for index in CartesianIndices((1:size(sokoban_map, 1), 1:size(sokoban_map, 2)))
        unit = sokoban_map[index] |> uppercase
        if unit == wall_unit
            continue
        end
        push!(reachable_units, index)
        reachable_dict[reachable_units[end]] = length(reachable_units)
        if unit == box_unit
            push!(box_init_pos, index)
        end
        if unit == goal_unit
            push!(goal_pos, index)
        end
        if unit == player_unit
            push!(player_pos, index)
        end
        if unit == box_goal
            push!(box_init_pos, index)
            push!(goal_pos, index)
        end
        if unit == player_goal
            push!(player_pos, index)
            push!(goal_pos, index)
        end
    end

    background = fill(Wall, size(sokoban_map))
    for e in reachable_units
        background[e] = Empty
    end

    Sokoban(background, box_init_pos, goal_pos, player_pos[1], reachable_units, reachable_dict)
end

Base.copy(xy::CartesianIndex) = CartesianIndex((xy.I...))
Base.copy(skb::Sokoban) = Sokoban(
    skb.Map, 
    copy(skb.box_pos), 
    skb.goal_pos, 
    copy(skb.player_pos), 
    skb.reachable_units, 
    skb.reachable_dict)

is_terminated(skb::Sokoban) = all(x -> x in skb.goal_pos, skb.box_pos)  # if all boxes are on the goal, the game ends
Base.show(io::IO, skb::Sokoban) = draw_symbols(skb)

function from_file(filename)
    lines = readlines(filename)
    skb_map = vcat([
        collect(line) |> x -> reshape(x, (1, length(x)))
    for line in lines]...)
    Sokoban(skb_map)
end

# Operations

function up!(skb::Sokoban)
    player = skb.player_pos
    if player[1] == 1 || skb.Map[player + UP] == Wall
        return
    end
    if player[1] > 2 && (player + UP) in skb.box_pos
        if (player + UP + UP) in skb.box_pos || skb.Map[player + UP + UP] == Wall
            return
        end
        idx = findfirst(isequal(player + UP), skb.box_pos)
        skb.box_pos[idx] += UP
    end
    skb.player_pos += UP
end


function down!(skb::Sokoban)
    lower = size(skb.Map, 1)
    player = skb.player_pos
    if player[1] == lower || skb.Map[player + DOWN] == Wall
        return
    end
    if player[1] < lower-1 && (player + DOWN) in skb.box_pos
        if (player + DOWN + DOWN) in skb.box_pos || skb.Map[player + DOWN + DOWN] == Wall
            return
        end
        idx = findfirst(isequal(player + DOWN), skb.box_pos)
        skb.box_pos[idx] += DOWN
    end
    skb.player_pos += DOWN
end

function left!(skb::Sokoban)
    player = skb.player_pos
    if player[2] == 1 || skb.Map[player + LEFT] == Wall
        return
    end
    if player[2] > 2 && (player + LEFT) in skb.box_pos
        if (player + LEFT + LEFT) in skb.box_pos || skb.Map[player + LEFT + LEFT] == Wall
            return
        end
        idx = findfirst(isequal(player + LEFT), skb.box_pos)
        skb.box_pos[idx] += LEFT
    end
    skb.player_pos += LEFT
end

function right!(skb::Sokoban)
    player = skb.player_pos
    bound = size(skb.Map, 2)
    if player[2] == bound || skb.Map[player + RIGHT] == Wall
        return
    end
    if player[2] < bound-1 && (player + RIGHT) in skb.box_pos
        if (player + RIGHT + RIGHT) in skb.box_pos || skb.Map[player + RIGHT + RIGHT] == Wall
            return
        end
        idx = findfirst(isequal(player + RIGHT), skb.box_pos)
        skb.box_pos[idx] += RIGHT
    end
    skb.player_pos += RIGHT
end

function draw_symbols(skb::Sokoban)
    game_map = copy(skb.Map)
    for box in skb.box_pos
        game_map[box] = Box
    end
    game_map[skb.player_pos] = Player
    for goal in skb.goal_pos
        if game_map[goal] == Empty
            game_map[goal] = Goal
        end
        if game_map[goal] == Player
            game_map[goal] = PlayerOnGoal
        end
        if game_map[goal] == Box
            game_map[goal] = BoxOnGoal
        end
    end
    
    for row in eachrow(game_map)
        for c in row
            print(type_to_symbol[c])
        end
        println()
    end
end

function step!(skb::Sokoban, action)
    if action == 1
        up!(skb)
    elseif action == 2
        right!(skb)
    elseif action == 3
        down!(skb)
    elseif action == 4
        left!(skb)
    end
end


# play it in the terminal
function interact!(skb::Sokoban)
    while true
        clear!()
        draw_symbols(skb)
        line = readline()
        if length(line) == 0
            buf = 'x'
        else
            buf = line[1]
        end

        if buf == 'w'
            up!(skb)
        elseif buf == 's'
            down!(skb)
        elseif buf == 'a'
            left!(skb)
        elseif buf == 'd'
            right!(skb)
        elseif buf == 'q'
            break
        end
    end
end