# basic token macros, mainly contain REST, SPLITTING, and CONTINUOUS
include("tokens.jl")

# config for muscic21 showing scores in musescore
include("config.jl")

# to read krn files
include("dataset.jl")
include("krnreader.jl")
include("krngenerator.jl")

# to read midi files
include("readfile.jl")
include("midi2vec.jl")
include("lookback.jl")



