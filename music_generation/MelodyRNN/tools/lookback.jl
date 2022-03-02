# the source file of lookback dataset preprocess in magenta is 
# `note_seq.encoder_decoder.LookbackEventSequenceEncoderDecoder`

using Flux: onehot

DEFAULT_LOOKBACK_TOKEN = CONTINUOUS_TOKEN  # means no event

"""
    Label a track to see whether it is a repetition from 2 bars ago, 1 bar ago, or neither.

This automatically label the scores, thus is a self-supervised method.

The encode is similar to that in magenta python implementation, which is a vector of:
    - one-hot representation of current note
    - one-hot representation of next step for 1-bar lookback
    - one-hot representation of next step for 2-bars lookback
    - [16th, 8th, 4th, half, whole] counter for position encoding
    - whether current step is repeating for [1 bar, 2 bars]

```Arguments
 - symbols: The vector of symbols 
 - upq: units per quarter
 - qpm: quarters per measure
```
"""
function lookback_input_encode(symbols, tokenset, position, upq; lookback_distance=[4upq, 8upq])
    rep = Float32.(onehot(symbols[position], tokenset))

    # get the symbol of lookback for next step
    for lookback_dist in lookback_distance
        lookback_pos = position - lookback_dist + 1

        # get token and use one-hot representation
        lookback_token = if lookback_pos <= 0
            DEFAULT_LOOKBACK_TOKEN
        else
            symbols[lookback_pos]
        end

        rep = vcat(rep, Float32.(onehot(lookback_token, tokenset)))
    end

    # position encoding
    pos_vec = map(4:-1:0) do note_dur
        measure_pos = (position+1) % (4upq)  # mod 4*upq (=4 quarters) and get relative offset

        # realign the offset
        if measure_pos == 0
            measure_pos = 4upq
        end

        # make relative pos to be united by 16th note
        if upq > 4
            measure_pos ÷= (upq÷4)
        end
        if upq < 4
            measure_pos *= (4÷upq)
        end
        
        measure_pos & (1 << note_dur)
    end

    rep = vcat(rep, pos_vec)

    # repeating flags
    for lookback_dist in lookback_distance
        lookback_pos = position - lookback_dist

        repeated = if lookback_pos <= 0 || symbols[lookback_pos] != symbols[position]
            0.0
        else
            1.0
        end

        push!(rep, repeated)
    end

    @assert (length(rep) == 3length(tokenset)+7) "encoding length errors"

    rep
end

"""
    Encode the label for token of given position. It contains the one-hot representation, and
lookback flags.
"""
function lookback_label_encode(symbols, position, upq; lookback_distance=[4upq, 8upq])
    # get lookback flags
    flags = map(lookback_distance) do lookback_dist
        lookback_pos = position - lookback_dist

        if lookback_pos <= 0 || symbols[lookback_pos] != symbols[position]
            0.0
        else
            1.0
        end
    end

    # seperate these two parts for easily computing loss
    (symbols[position], flags)
end


function generate_lookback_dataset(songs, seq_len, upq)

    xs = []
    ys_note = []
    ys_lookback = []

    # constrcut tokenset
    tokenset = Set()
    push!(tokenset, SPLITTING_TOKEN)
    for song in songs
        union!(tokenset, song)
    end
    tokenset = collect(tokenset)

    # extract xs and ys
    for song in songs
        padded_song = vcat(
            fill(SPLITTING_TOKEN, seq_len),
            song,
            fill(SPLITTING_TOKEN, seq_len)
        )

        # encode input vectors
        inputs = map(1:length(padded_song)-1) do pos
            lookback_input_encode(padded_song, tokenset, pos, upq)
        end

        # pack input and label
        for pos in 1:length(inputs)-seq_len+1
            # [a,b,c,d,...], use [a,b,c] to predict `d`
            push!(xs, inputs[pos:pos+seq_len-1])

            note, lookback = lookback_label_encode(padded_song, pos+seq_len, upq)
            push!(ys_note, note)
            push!(ys_lookback, lookback)
        end
    end

    # create matrix representation
    xs = map(1:seq_len) do idx
        # combine the `i`th note vectors into a matrix
        seq_at_idx = map(xs) do seq
            seq[idx]
        end

        hcat(seq_at_idx...)
    end

    # in the test time, using hcat for onehot vectors had met some problems
    # thus here use Flux built in `onehotbatch` function
    ys_note = onehotbatch(ys_note, tokenset)
    ys_lookback = hcat(ys_lookback...)

    (xs, ys_note, ys_lookback), tokenset
end
