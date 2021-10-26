using Statistics: var, mean

function AdaIn(c, s)
    # WHCN
    c′ = (c .- mean(c, dims=(1,2))) ./ e_std(c, dims=(1,2))
    c′ .* e_std(s, dims=(1,2)) .+ mean(s, dims=(1,2))
end

function e_std(x; dims)
    ϵ::Float32 = 1e-2
    sqrt.(var(x, dims=dims) .+ ϵ)
end