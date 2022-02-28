# this file provides model for `krnreader.jl`
# The structure is similar to base melodyrnn, which needs a note ranging,
# and take midi event in one-hot vectors as input. 

using Flux

mutable struct KrnModel
    rnn
    mlp
end

Flux.params(m::KrnModel) = Flux.params(m.rnn, m.mlp)

function model_for_krn(note_dim, hidden_dim)
    KrnModel(
        LSTM(note_dim, hidden_dim),
        Chain(
            Dropout(0.2),
            Dense(hidden_dim, note_dim)
        )
    )
end

function to_device(m::KrnModel, device)
    m.rnn = device(m.rnn)
    m.mlp = device(m.mlp)
    m
end

function (m::KrnModel)(seq)
    # reset inner state
    Flux.reset!(m)

    # push series into model, and return the last one
    hiddens = [m.rnn(x) for x in seq]
    m.mlp(hiddens[end])
end
