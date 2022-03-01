using JLD2

include("../tools/tokens.jl")
include("../tools/krngenerator.jl")


function load_and_generate(model_path)
    @load model_path model token_map seq_len

    generate_melody(model, token_map; seq_len=seq_len)
end


load_and_generate("deutschl_model.jld2")
