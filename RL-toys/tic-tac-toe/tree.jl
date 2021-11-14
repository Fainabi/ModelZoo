abstract type AbstractTreeNode end
abstract type AbstractTree end

Base.@kwdef mutable struct MonteCarloTreeNode{V} <: AbstractTreeNode
    Parent::Vector{Int} = Int[]
    Children::Vector{Int} = Int[]
    val::Vector{V}  # for constructing a reference
    id::Int
    visited::Int
    MonteCarloTreeNode{V}(id) where V = new{V}(Int[], Int[], V[0.0], id, 0)
    MonteCarloTreeNode(id) = MonteCarloTreeNode{Float64}(id)
end
function add_child!(parent::AbstractTreeNode, child::AbstractTreeNode)
    push!(parent.Children, child.id)
    push!(child.Parent, parent.id)
end
function reward(node::MonteCarloTreeNode)
    node.val[]  # dereference
end


mutable struct MonteCarloTree{V} <: AbstractTree
    id_map::Dict{Int, MonteCarloTreeNode{V}}
    root::Union{Int, Nothing}
    size::Int
    now_id::Int
    total_visited::Int
    function MonteCarloTree{V}() where V
        new(Dict{Int, MonteCarloTreeNode{V}}(), nothing, 0, 0, 0)
    end
    function MonteCarloTree(root::MonteCarloTreeNode{T}) where T
        dict = Dict{Int, MonteCarloTreeNode{T}}(
            root.id => root
        )
        new{T}(dict, root.id, 1, root.id+1, 0)
    end
end
function Base.empty!(tree::MonteCarloTree)
    tree.root = nothing
    tree.size = 0
    empty!(tree.id_map)
    tree.now_id = 0
    tree.total_visited = 0
end

(tree::MonteCarloTree)(id::Int) = tree.id_map[id]
function spawn!(tree::MonteCarloTree, id::Int)
    parent = tree(id)
    child = MonteCarloTreeNode(tree.now_id)
    tree.now_id += 1

    add_child!(parent, child)
    tree.id_map[child.id] = child
end

# specific child id
function spawn!(tree::MonteCarloTree, pid::Int, cid::Int)
    parent = tree(pid)
    child = MonteCarloTreeNode(cid)
    if cid ∉ parent.Children
        add_child!(parent, child)
        tree.id_map[child.id] = child
    end
end
function spawn!(tree::MonteCarloTree, pid::Int=1)  # set root
    if isempty(tree.id_map)
        node = MonteCarloTreeNode(pid)
        tree.id_map[node.id] = node
    end
end
# spawn!(tree::MonteCarloTree, node::MonteCarloTreeNode) = spawn!(tree, node.id)
function visit!(tree::MonteCarloTree, id::Int)
    node = tree(id)
    node.visited += 1
    tree.total_visited += 1
end
function visited(tree::MonteCarloTree, ids::Vector{Int})
    map(ids) do id
        visited(tree, id)
    end
end
function visited(tree::MonteCarloTree, id::Int)
    if haskey(tree.id_map, id)
        tree.id_map[id].visited
    else
        0
    end
end

isnode(tree::MonteCarloTree, id) = haskey(tree.id_map, id)

