# midi to vectors

using MusicManipulations

#=
A midi file can be read by using MIDI pacakge, and the structure is like:

MIDI.load(filename) --> MIDIFile(format, tpq, tracks)
                                                ||
                                                ||
                        MIDITrack(events) <-- Vector{MIDITrack}
                                    ||
                                    ||
                        Vector{TrackEvent} -->  MIDI.NoteOnEvent, 
                                                MIDI.NoteOffEvent, 
                                                MIDI.TrackNameEvent ...
=#

"""
    Remove the non-note event.
"""
function track_quantize(track::MIDI.MIDITrack, grid)
    qnotes = quantize(getnotes(track), grid)
    notes_track = MIDITrack()
    addnotes!(notes_track, qnotes)

    notes_track
end

"""
    Remove the track which has overlapping of notes.
"""
function filter_melody(track::MIDI.MIDITrack; allow_overlap=false)
    # A note contains `channel`, `duration`, `pitch`, `position` and `velocity`.
    # We only need `duration`, `pitch` and ``position.

    notes = getnotes(track)

    # sort the notes by position
    sorted_notes = sort(notes.notes; by=(n -> n.position))

    last_note = nothing
    flattened_notes = Note[]

    # travel the notes to detect overlapping
    for note in sorted_notes
        if !isnothing(last_note)

            # meet overlapping
            if last_note.position + last_note.duration > note.position
                # perfectly overlap
                if last_note.position == note.position && last_note.duration == note.duration
                    continue
                end

                if !allow_overlap
                    return Note[]
                end

                error("Not implement notes allow overlapping")
            end
        end

        push!(flattened_notes, note)
        last_note = note
    end

    flattened_notes
end

"""
    Flatten the distincted notes into one-hot representation
"""
function notes_to_symbols(notes, grid, quarter_duration)
    tokens = []

    grid = sort(collect(grid))
    unit = grid[2] * quarter_duration  # the second smaller one is unit 

    # track latest note and tick
    # start near note[1].position to reduce blank rest notes
    tick = notes[1].position - (notes[1].position % (4quarter_duration))

    for note in notes
        # if has gap, fill it with rest symbols
        # This may cause long duration rest, one needs to splitting it into snippets
        if tick < note.position
            push!(tokens, REST_TOKEN)

            token_nums = (note.position - tick) ÷ unit
            for _ in 2:token_nums
                push!(tokens, CONTINUOUS_TOKEN)
            end

            # after quantize, the position must be divided by the unit
            tick = note.position
        end

        # push the note
        push!(tokens, note.pitch)
        token_nums = note.duration ÷ unit
        for _ in 2:token_nums-1
            push!(tokens, CONTINUOUS_TOKEN)
        end

        tick += note.duration
    end

    tokens
end


"""
    Extract all tracks contain only melodies, and filter out those
have multiple notes at any time.
"""
function extract_melody_tracks(midifile::MIDIFile, config::MelodyRNNConfig)
    melody_tracks = []
    grid = config.event_config.grid
    
    for track in midifile.tracks
        # quantize the track
        quantized_track = track_quantize(track, grid)

        # check overlapping
        notes = filter_melody(quantized_track)

        isempty(notes) && continue

        # change into representation
        symbols = notes_to_symbols(notes, grid, midifile.tpq)

        # needs checking if is an accompany track
        push!(melody_tracks, symbols)
    end

    melody_tracks
end


"""
    Construct a pianoroll representation for midifile. It is suitable for track
with multiple channels.
"""
function midifile_to_pianoroll(midifile::MIDIFile, config::MelodyRNNConfig)
    tracks = midifile.tracks
    grid = config.event_config.grid
    
    pianorolls = map(tracks) do track
        notes = getnotes(track)
        
        # segment the notes into grids
        if config.event_config.hparams["segment"]
            notes = segment(notes, grid)
            #timeseries(segmented_notes, :pitch, minimum, grid)
        end

        end_of_note_tick = map(notes.notes) do note
            note.position + note.duration
        end |> maximum
        minimal_duration = map(notes.notes) do note
            note.duration
        end |> minimum

        number_of_steps = (end_of_note_tick / minimal_duration) |> Int∘ceil
        number_of_notes_candidate = (config.max_note - config.min_note) + 1
        roll = zeros(Bool, number_of_notes_candidate, number_of_steps)
        for note in notes.notes
            # if it has rest before, push it back to extend duration
            pos = (note.position / minimal_duration) |> Int∘floor
            dur = (note.duration / minimal_duration) |> Int∘ceil
            pitch = note.pitch - config.min_note + 1
            roll[pitch, pos+1:pos+dur] .= true
        end

        roll
    end
end

