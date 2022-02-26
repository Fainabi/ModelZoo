# this file provides model for `krnreader.jl`

using Flux

struct KrnModel
    rnn
    mlp
end

# Flux.trainable(m::KrnModel) = (Flux.trainable(m.rnn), Flux.trainable(m.mlp))
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

function (m::KrnModel)(seq)
    # reset inner state
    Flux.reset!(m)

    # push series into model, and return the last one
    hiddens = [m.rnn(x) for x in seq]
    m.mlp(hiddens[end])
end
