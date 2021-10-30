function clear!()
    Base.run(`cmd /c cls`)
end

function get_keypress()
    ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid}, Int32), stdin.handle, true)
    c = read(stdin, Char)
    ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid}, Int32), stdin.handle, false)
    c
end

function binomial_order(n, k)
    pos_state = Dict()
    posistions = []

    pos = collect(1:k)
    space_size = binomial(n, k)
    if space_size > 100000
        @warn "This state space is too big for DP ($space_size), consider non DP methods. "
    end
    for i in 1:space_size
        pos_state[copy(pos)] = i
        push!(posistions, copy(pos))
        next_pos!(n, pos)
    end

    posistions, pos_state
end

function next_pos!(n, pos)
    for k = 1:length(pos)
        if pos[end-k+1] == n-k+1
            continue
        end
        pos[end-k+1] += 1
        for i = length(pos)-k+2:length(pos)
            pos[i] = pos[i-1] + 1
        end
        break
    end
end