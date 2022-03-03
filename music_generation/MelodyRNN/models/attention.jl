# Attention Melody RNN
## In the attention model, we track `n` steps of rnn output to create attention coefficients
## In the blog (https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn/),
## it is suggested that push the combined output together with input of next step ,
## as a new input, and pass through a new layer.


using Flux
using Flux.Zygote

"""
    AttentionLayer

Compute attention coefficients of input states, and do linear combination of that.
"""
mutable struct AttentionLayer
    W1
    W2
    v
    function AttentionLayer(hdim, cdim, vdim)
        new(
            Dense(hdim, vdim),
            Dense(cdim, vdim),
            Dense(vdim, 1)
        )
    end
end

Flux.params(al::AttentionLayer) = Flux.params(al.W1, al.W2, al.v)

"""
    Compute attention of give `h`s and inner state of rnn `c`
"""
function (al::AttentionLayer)(hs, c)
    # this design doesnot allow batch training
    ui = al.v * tanh.(al.W1 * hs .+ al.W2 * c)
    ai = softmax(ui)

    hs * ai
end

"""
    CircularVector provides tracking of sequential data.
"""
mutable struct CircularVector
    position
    records
    CircularVector{T=Float32}(n::Integer, vec_len::Integer) = new(1, zeros(T, vec_len, n))
end

function Base.push!(cv::CircularVector, vec)
    cv.records[:, cv.position] .= vec

    cv.position += 1
    if cv.position > size(cv.records, 2)
        cv.position = 1
    end
end

function reset!(cv::CircularVector)
    cv.position = 1
    cv.records .= zero(cv.records)
end


"""
    AttentionMelodyRNN
"""
mutable struct AttentionMelodyRNN
    rnn
    input_layer
    output_layer
    attention_layer::AttentionLayer
    record::CircularVector
    function AttentionMelodyRNN(rnn_dims::Pair, input_layer, output_layer, attention_layer, record)
        new(
            LSTM(rnn_dims[1], rnn_dims[2]),
            input_layer,
            output_layer,
            attention_layer,
            record
        )
    end
end


function (am::AttentionMelodyRNN)(seq)
    Flux.reset!(am.rnn)

    # handle inputs    
    inputs = am.input_layer.(seq)
    state = [am.rnn(x) for x in inputs]

    # compute attention
    ht = am.attention_layer(am.record.records, state[end])

    # here no concatenation
    am.output_layer(ht)
end
