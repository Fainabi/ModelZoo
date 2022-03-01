# lookback melodyrnn model
# From the blog (https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn/),
#   we know that Lookback RNN built on the base melodyRNN, 
#   take the former 1-2 bars of notes, flag of repeating 
#   and encode of current position as the additional inputs.
# The corresponded labels are added as well.

using Flux

# generatl structure of lookback rnn is same to the basic rnn
# the dataset construction imitates that in magenta, but model
# and data processing have some differences.

"""
    Lookback melody rnn has a rnn to process sequential data,
and two multiple layer perceptrons, for generating the next note
and the lookback flags.
"""
mutable struct LookbackMelodyRNN
    rnn
    mlp_note
    mlp_lookback
end

Flux.params(m::LookbackMelodyRNN) = (m.rnn, m.mlp_note, m.mlp_lookback)

function construct_lookback_rnn(in_dim, hidden_dim, note_dim)
    LookbackMelodyRNN(
        LSTM(in_dim + 7, hidden_dim),       # rnn
        Chain(
            Dropout(0.2),
            Dense(hidden_dim, note_dim),    # mlp note
        ),
        Chain(
            Dropout(0.2),
            Dense(hidden_dim, 2)            # mlp lookback
        )
    )
end

function to_device(m::LookbackMelodyRNN, device)
    m.rnn = device(m.rnn)
    m.mlp_note = device(m.mlp_note)
    m.mlp_lookback = device(m.mlp_lookback)

    m
end


function (m::LookbackMelodyRNN)(seq)
    Flux.reset!(m.rnn)

    hiddens = [m.rnn(x) for x in seq]
    (m.mlp_note(hiddens), m.mlp_lookback(hiddens))    
end
