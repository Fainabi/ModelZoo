using JLD2
using Flux: onehot, onehotbatch

include("event.jl")
include("modes.jl")

function erase_continuous_symbol!(melody)
    for (idx, token) in enumerate(melody)
        if token == CONTINUOUS_TOKEN || token == CHORD_CONTINUOUS_TOKEN
            melody[idx] = melody[idx-1]
        end
    end

    melody
end

function erase_pauses!(bar_of_melody; default_token)
    if bar_of_melody[1] == REST_TOKEN || bar_of_melody[1] == CHORD_REST_TOKEN
        idx = findfirst(bar_of_melody) do note
            (note != REST_TOKEN) && (note != CHORD_REST_TOKEN)
        end

        if isnothing(idx)
            bar_of_melody .= default_token
        else
            bar_of_melody[1:idx] .= bar_of_melody[idx]
        end
    end

    for (idx, token) in enumerate(bar_of_melody)
        if token == REST_TOKEN || token == CHORD_REST_TOKEN
            bar_of_melody[idx] = bar_of_melody[idx-1]
        end
    end

    bar_of_melody
end

function transpose_to_c(melodies, chords, key)
    melodies = map(melodies) do m 
        m .- KEYS[key]
    end

    chords = map(chords) do chord
        # length of string is >= 2
        root = KEYS[string(chord[1])]

        mark_idx = 2
        if chord[2] == '#'
            root += 1
            mark_idx += 1
        elseif chord[2] == 'b'
            root -= 1
            mark_idx += 1
        end
        
        # get the chords is major or minor, the diminished chord is regarded as minor
        # 0 for major chords, 1 for minor chords
        chord_type = if chord[mark_idx] == '7' || chord[mark_idx] == 'M' || chord[mark_idx] == '+'
            0
        else
            1
        end

        [root, chord_type]
    end

    melodies, chords
end

"""
    The melody is built with 16th notes as units, so every measure has 16 units.
"""
function truncate_measures(melody)
    unit_num = length(melody)
    unit_num -= unit_num % 16

    melody_mat = reshape(melody[1:unit_num], 16, :)
    [melody_mat[:, col] for col in 1:size(melody_mat, 2)]
end

function transpose_to(dataset, target_tonic)
    map(dataset) do ds
        melodies, chords = ds
        melodies = map(melodies) do m
            m .+ target_tonic
        end
        chords = map(chords) do c
            [(c[1] + target_tonic) % 12, c[2]]
        end

        melodies, chords
    end
end

"""
    For every piece, transpose it to generate 12 snippets of different keys.
"""
function augment_dataset(dataset)
    new_dataset = []

    for tonic in 0:12
        new_dataset = vcat(new_dataset, transpose_to(dataset, tonic))
    end

    new_dataset
end

function sequences_to_matrix(seq)
    seq = vcat(seq...)
    seq = onehotbatch(seq, DEFAULT_PITCH_RANGE)
    seq = reshape(seq, length(DEFAULT_PITCH_RANGE), 16, :)
    seq = Float32.(seq)
end

function preprocess_dataset(filename="theorytab.jld2")
    # load the dataset
    JLD2.@load filename melodies chords keys

    # one beat has four units
    # unit is 16th note here
    min_dur = 0.25

    # in the origin paper, the authors augment dataset by transposing the songs
    # here, we skip the step

    # the paper also replace the pauses with the former notes, and the pauses start at a bar with the following notes
    erase_continuous_symbol!.(melodies)
    erase_continuous_symbol!.(chords)

    # get the measure slices
    melodies = map(melodies, keys) do melody, key
        no_pauses_song = erase_pauses!(melody; default_token=CENTER_C + KEYS[key])

        truncate_measures(no_pauses_song)
    end

    chords = map(chords, keys) do chord, key
        no_pauses_chord = erase_pauses!(chord; default_token=string(key, "M"))

        # every bar has only one chord
        map(truncate_measures(no_pauses_chord)) do chords
            # the first chord to be set for that bar
            first(chords)
        end
    end

    # transpose 
    dataset = map(melodies, chords, keys) do ms, cs, k
        ms, cs = transpose_to_c(ms, cs, k)

        # realign 
        common_len = min(length(ms), length(cs))
        ms[1:common_len], cs[1:common_len]
    end

    # augmentation
    dataset = augment_dataset(dataset)

    # for discriminator, (conditions, X) => 1, is corresponded to 
    # the continous pair in the dataset. Here create such labeled 1 dataset
    real_dataset_conditions_2d = []
    real_dataset_conditions_1d = []
    real_dataset_inputs = []
    for (bars, chords) in dataset
        for idx in 1:length(bars)-1
            push!(real_dataset_conditions_1d, chords[idx+1])
            push!(real_dataset_conditions_2d, bars[idx])
            push!(real_dataset_inputs, bars[idx+1])
        end
    end

    # one hot encoding

    real_dataset_inputs = sequences_to_matrix(real_dataset_inputs)
    real_dataset_conditions_2d = sequences_to_matrix(real_dataset_conditions_2d)
    real_dataset_conditions_1d = map(real_dataset_conditions_1d) do pair
        vcat(onehot(pair[1], 0:11), pair[2])
    end
    real_dataset_conditions_1d = Float32.(hcat(real_dataset_conditions_1d...))

    (real_dataset_inputs, real_dataset_conditions_2d, real_dataset_conditions_1d)
end
