include("process.jl")

using JLD2
using ProgressMeter

function build_dataset_file(song_num=1000; dataset_path = "theorytab/", min_dur=0.25)
    # this preprocess will not contain chords

    song_cnt = 0
    song_series = []
    song_keys = []
    song_chords = []
    p = Progress(song_num, 1)

    for (root, _, files) in walkdir(dataset_path), file in files
        if song_cnt > song_num
            break
        end

        # for some unimplemented version of forms and error data, ignore them
        song_data = try
            read_song_data(joinpath(root, file); chords=true)
        catch
            continue
        end

        if isnothing(song_data)
            continue
        end
        song_tokens, dur, song_key, song_chord = song_data

        # to make the dataset aligend, thus filter out songs whose duration is too small
        if dur != min_dur
            continue
        end

        push!(song_series, song_tokens)
        push!(song_keys, song_key)
        push!(song_chords, song_chord)

        song_cnt += 1
        next!(p)
    end

    JLD2.@save "theorytab.jld2" melodies=song_series chords=song_chords keys=song_keys
end
