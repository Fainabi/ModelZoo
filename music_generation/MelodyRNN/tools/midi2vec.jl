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

function midifile_event_series(midifile::MIDIFile, config::MelodyRNNConfig)
    tracks = midifile.tracks
    grid = config.event_config.grid

    map(tracks) do track
        notes = getnotes(track)

        
    end
end
