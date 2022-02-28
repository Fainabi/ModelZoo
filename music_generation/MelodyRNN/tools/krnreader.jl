import PyCall
music21 = PyCall.pyimport("music21")

using Flux: onehotbatch, onehot

# some default configs
ACCEPTABLE_DURATIONS = [
    1//4,
    1//2,
    3//4,
    1,    # a quarter note
    3//2,
    2,
    3,
    4, 
]


function load_dataset(dataset_path)
    println("Reading krn files in ", dataset_path)
    songs = []
    for item in readdir(dataset_path)
        item_path = joinpath(dataset_path, item)
        # needs files
        !isfile(item_path) && continue

        # needs .krn files
        !endswith(item, ".krn") && continue

        # use music21 to load file
        song = music21.converter.parse(item_path)
        push!(songs, song)
    end

    # returns a list of PyObject of musci21
    songs
end

function has_acceptable_durations(song, acceptable_durations)
    all(song.flat.notesAndRests) do note
        note.duration.quarterLength ∈ acceptable_durations
    end
end

# target key is Cmaj
function key_transpose(song::PyCall.PyObject)
    # get the key of the song snippet
    parts = song.getElementsByClass(music21.stream.Part)
    first_part_measures = parts[1].getElementsByClass(music21.stream.Measure)
    key = first_part_measures[1][5]

    # not get a key and analyze it
    if !PyCall.py"isinstance"(key, music21.key.Key)
        key = song.analyze("key")
    end

    # get interval
    interval = if key.mode == "major"
        music21.interval.Interval(key.tonic, music21.pitch.Pitch("C"))
    elseif key.mode == "minor"
        music21.interval.Interval(key.tonic, music21.pitch.Pitch("A"))
    end

    # transpose
    song.transpose(interval)
end


function encode_song(song; time_step=1//4)
    # take the midi range representation, 0-127 for notes, 128 for continuous mark
    # pitch = 60, duration = 1.0 => [60, 128, 128, 128]

    encoded_song = []

    for event in song.flat.notesAndRests
        symbol = if PyCall.py"isinstance"(event, music21.note.Note)
            event.pitch.midi
        elseif PyCall.py"isinstance"(event, music21.note.Rest)
            129  # for rest
        end

        steps = (event.duration.quarterLength / time_step) |> Int∘floor
        for step in 1:steps
            encoded_symbol = if step == 1
                symbol
            else
                128
            end

            push!(encoded_song, encoded_symbol)
        end
    end

    encoded_song
end

function preprocess(songs)
    filtered_songs = []
    for song in songs
        # filter out the songs have nonacceptable durations
        !has_acceptable_durations(song, ACCEPTABLE_DURATIONS) && continue

        # transpose into Cmaj
        transposed_song = key_transpose(song)

        # encode into representation
        encoded_song = encode_song(transposed_song)

        push!(filtered_songs, encoded_song)
    end

    filtered_songs
end

"""
    Generate the training sequences with input songs. These songs need preprocessing.
    
`sequence_length` provides the length of every input sequences in dataset,
and each song with be padded with splitting token (255) whose length is equal to sequence length.
"""
function generate_training_sequences(songs, sequence_length)

    # create subsequences as training set
    xs = []
    ys = []
    token_set = Set()

    for song in songs
        padded_song = vcat(
            fill(255, sequence_length),
            song,
            fill(255, sequence_length)
        )

        for idx in 1:length(padded_song)-sequence_length
            push!(xs, padded_song[idx:idx+sequence_length-1])
            push!(ys, padded_song[idx+sequence_length])
        end

        # fill the token set
        token_set = union(token_set, Set(padded_song))
    end

    # create mapping
    token_set = collect(token_set)
    
    # create onehot encodings
    # since every sequence in xs are aligned, we could directly construct
    # a tensor and apply gpu without realigning the dataset

    # xs = map(xs) do seq
    #     [Float32.(onehot(token, token_set)) for token in seq]
    # end
    # ys = [onehot(y, token_set) for y in ys]

    xs = hcat(xs...)
    xs = map(1:size(xs, 1)) do r
        # get rows of matrix, corresponding to the `r`th step of rnn
        Float32.(onehotbatch(xs[r, :], token_set))
    end
    ys = onehotbatch(ys, token_set)

    # return dataset, token_set
    (xs, ys), token_set
end
