include("../models/config.jl")
include("../models/lookback.jl")

include("../tools/tokens.jl")
include("../tools/lookback.jl")

using JLD2

function lookback_load_and_generate(modelpath)
    @load modelpath model tokenset seq_len

    notes = generate_lookback_melody(model, tokenset, [60, CONTINUOUS_TOKEN, CONTINUOUS_TOKEN, CONTINUOUS_TOKEN]; 
        seq_len=seq_len, num_steps=300, lookback_weights=[0, 0])

    println("Generated ", length(notes), " notes")

    file = write_lookback_melody(notes, 4)

    notes, file
end

lookback_load_and_generate("lookback.jld2")
