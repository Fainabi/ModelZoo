using JLD2

include("../tools/krngenerator.jl")


function load_and_generate(model_path)
    @load model_path model token_map

    generate_melody(model, token_map; seq_len=3)
end


load_and_generate("deutschl_model.jld2")
