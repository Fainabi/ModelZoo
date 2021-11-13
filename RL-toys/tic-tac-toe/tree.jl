abstract type AbstractTreeNode end
abstract type AbstractTree end

Base.@kwdef mutable struct MonteCarloTreeNode{V} <: AbstractTreeNode
    Parent::Union{Int, Nothing} = nothing
    Children::Vector{Int} = Int[]
    val::V
    id::Int
    visited::Int
    MonteCarloTreeNode(id) = new{Int}(nothing, Int[], 0, id, 0)
end
function add_child!(parent::AbstractTreeNode, child::AbstractTreeNode)
    push!(parent.Children, child.id)
    child.Parent = parent.id
end
function reward(node::MonteCarloTreeNode)
    if node.visited == 0
        0.0
    else
        node.val / node.visited
    end
end


mutable struct MonteCarloTree{V} <: AbstractTree
    id_map::Dict{Int, MonteCarloTreeNode{V}}
    root::Union{Int, Nothing}
    size::Int
    now_id::Int
    function MonteCarloTree{V}() where V
        new(Dict{Int, MonteCarloTreeNode{V}}(), nothing, 0, 0)
    end
    function MonteCarloTree(root::MonteCarloTreeNode{T}) where T
        dict = Dict{Int, MonteCarloTreeNode{T}}(
            root.id => root
        )
        new{T}(dict, root.id, 1, root.id+1)
    end
end
(tree::MonteCarloTree)(id::Int) = tree.id_map[id]
function spawn!(tree::MonteCarloTree, id::Int)
    parent = tree(id)
    child = MonteCarloTreeNode(tree.now_id)
    tree.now_id += 1

    add_child!(parent, child)
    tree.id_map[child.id] = child
end
# spawn!(tree::MonteCarloTree, node::MonteCarloTreeNode) = spawn!(tree, node.id)
function visit!(tree::MonteCarloTree, id::Int)
    parent = tree(id)
    parent.visited += 1
end

