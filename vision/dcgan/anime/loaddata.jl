using BSON
using Images

# about 1 gigabytes data to load

function loaddata()
    path = joinpath(@__DIR__, "data")
    data = zeros(Float32, 64, 64, 3, 21551)
    for idx in 1:21551
        img = float32.(Images.load(joinpath(path, "$idx.png")))
        data[:, :, :, idx] .= permutedims(channelview(img), [3,2,1])
    end
    BSON.@save joinpath(path, "data.bson") data
end
