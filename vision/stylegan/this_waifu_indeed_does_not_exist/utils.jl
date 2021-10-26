using Statistics: var, mean, std

function AdaIN(c, s)
    # WHCN
    len = length(s)  # 1024
    s = reshape(s, 1, 1, :, 1)
    
    c′ = (c .- mean(c, dims=(1,2))) ./ e_std(c, dims=(1,2))
    # c′ .* e_std(s, dims=(1,2)) .+ mean(s, dims=(1,2))
    c′ .* s[:,:, 1:Int(len/2) ,:] .+ s[:,:, Int(len/2)+1:end, :]
end

function IN(x)
    (x .- mean(x, dims=(1,2))) ./ e_std(x, dims=(1,2))
end

function e_std(x; dims)
    ϵ::Float32 = 1e-8
    sqrt.(var(x, dims=dims) .+ ϵ)
end