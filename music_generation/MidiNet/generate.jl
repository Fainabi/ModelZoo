include("event.jl")
include("modes.jl")

using JLD2
using Flux: DataLoader, glorot_normal, onehotbatch, onecold
using MIDI
using StatsBase

function fill_with_continuous_token!(notes)
    for idx in length(notes):-1:2
        if notes[idx] == notes[idx-1]
            notes[idx] = CONTINUOUS_TOKEN
        end
    end
end


function matrix_to_notes(mat)
    mat = reshape(mat, length(DEFAULT_PITCH_RANGE), :)
    # notes = onecold(mat, DEFAULT_PITCH_RANGE)
    notes = [StatsBase.sample(DEFAULT_PITCH_RANGE, StatsBase.Weights(mat[:, i])) for i in 1:size(mat, 2)]

    notes = reshape(notes, 16, :)
    notes = [notes[:, i] for i in 1:size(notes, 2)]
    fill_with_continuous_token!.(notes)

    notes
end

function notes_to_midinotes(notes, tpu)
    midinotes = Note[]

    now_pitch = nothing
    now_dur = 1
    now_offset = 0
    for note in notes
        if note == CONTINUOUS_TOKEN
            now_dur += 1
        else
            if !isnothing(now_pitch)
                push!(midinotes, Note(now_pitch, 96, now_offset, now_dur*tpu))
                now_offset += now_dur*tpu
            end

            now_pitch = note
            now_dur = 1
        end
    end

    # the last note
    push!(midinotes, Note(now_pitch, 96, now_offset, now_dur*tpu))

    Notes(midinotes)
end

function chords_to_midichords(chords, tpq)
    midinotes = Note[]

    for (idx, chord) in enumerate(chords)
        position = 4 * (idx-1) * tpq

        chord_root, chord_type = chord

        if chord_root > 4
            chord_root -= 12
        end
        
        root_pitch = CENTER_C - 12 + chord_root

        push!(midinotes, Note(root_pitch, 96, position, 4tpq))
        push!(midinotes, Note(root_pitch + 7, 96, position, 4tpq))
        if chord_type == 0
            push!(midinotes, Note(root_pitch + 4, 96, position, 4tpq))
        else
            push!(midinotes, Note(root_pitch + 3, 96, position, 4tpq))
        end
    end

    Notes(midinotes)
end

function generate_midi(num=1; bars=8)

    # load model and dataset
    JLD2.@load "dataset.jld2" cond2d
    JLD2.@load "model.jld2" gen

    
    # randomly choose priming melodies
    cond2d = DataLoader(cond2d, ; batchsize=num, shuffle=true) |> first
    
    generated_matrices = []
    midi_files = map(1:num) do _
        file = MIDIFile()
        melody_track = MIDITrack()
        chord_track = MIDITrack()

        addtrackname!(melody_track, "melody")
        addtrackname!(chord_track, "chord")

        push!(file.tracks, melody_track)
        push!(file.tracks, chord_track)

        file
    end
    midi_notes = map(1:num) do _
        []
    end
    midi_chords = map(1:num) do _
        []
    end

    tpq = 960
    tpu = tpq ÷ 4
    for _ in 1:bars
        cond2d = if isempty(generated_matrices)
            cond2d
        else
            generated_matrices[end]
        end
        z = glorot_normal(100, num)

        # randomly choose chords
        chord_root = rand(0:11, num)
        chord_type = rand(0:1, 1, num)
        cond1d = onehotbatch(chord_root, 0:11)
        cond1d = vcat(cond1d, chord_type)
        cond1d = Float32.(cond1d)
        
        # generate matrix
        mat = gen(z, cond1d, cond2d)
        push!(generated_matrices, mat)

        # transfer matrix into midi notes
        melody = matrix_to_notes(mat)

        for (idx, notes) in enumerate(melody)
            midi_notes[idx] = vcat(midi_notes[idx], notes)
        end

        for idx in 1:num
            push!(midi_chords[idx], [chord_root[idx], chord_type[idx]])
        end
    end

    # write midi files
    for idx in 1:num
        file = midi_files[idx]
        melody = midi_notes[idx]
        notes = notes_to_midinotes(melody, tpu)
        chords = chords_to_midichords(midi_chords[idx], tpq)

        addnotes!(file.tracks[1], notes)
        addnotes!(file.tracks[2], chords)
        writeMIDIFile(string("midis/", idx, ".mid"), file)
    end
end
