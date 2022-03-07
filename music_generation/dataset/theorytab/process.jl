# This file provides functions to build dataset with crawled theorytab songs

# Among the crawled data, the notes and chords are stored in either `xmlData` or `jsonData` attribute.

using JSON3
using EzXML

include("modes.jl")
include("event.jl")


function normalize_pitch_string(ps)
    sharps = length(findall('#', ps))
    flats = length(findall('b', ps))

    ps = replace(ps, "#"=>"", "b"=>"")
    string(ps, fill('#', sharps)..., fill('b', flats)...)
end

"""
    extract the MIDI number of pitch, by given that note, meters, and keys
"""
function parse_pitch(note, song_keys)
    note_sd = normalize_pitch_string(note.sd)
    pitch_num = parse(Int, note_sd[1])
    
    # find the key closed and before the note
    key_idx = findlast(song_keys) do sk
        sk.beat <= note.beat
    end
    note_key = song_keys[key_idx]
    scale, tonic = note_key.scale, note_key.tonic

    scale = MODES[scale]
    tonic = CENTER_C + KEYS[tonic]

    # get pitch of note
    pitch = 12*note.octave + tonic + scale[pitch_num]
    for ch in note_sd[2:end]
        if ch == '#'
            pitch += 1
        elseif  ch == 'b'
            pitch -= 1
        end
    end

    pitch
end


function parse_chord(chord, song_keys; max_chord_type=7)
    # chord type is in [5(for triad), 7, 9, 11, 13], the traditional 3th repeated chords
    # inversions are [0(None), 1st, 2nd, 3rd], here we don't care about it
    # beat, root, duration are numbers
    # The `adds`, `omits`, `alterations`, and `suspentions` attributes are omitted here.
    #
    # `borrowed`, `applied`, and `root` determine the absolute root pitch of the chord

    # is a rest
    if chord.root == 0
        return CHORD_REST_TOKEN
    end


    key_idx = findlast(song_keys) do sk
        sk.beat <= chord.beat
    end
    chord_applied = (iszero(chord.applied)) ? 1 : chord.applied
    chord_key = song_keys[key_idx]
    scale, tonic = chord_key.scale, chord_key.tonic

    scale = MODES[scale]    
    tonic = KEYS[tonic]     # no CENTER_C offset here

    # chord.root is an integer, and means would not be out of scale range
    applied_tonic = (tonic + scale[chord.root]) % 12    # chord.root is one-based
    scale_name = if !isnothing(chord.borrowed) && !isempty(chord.borrowed)
        chord.borrowed
    else
        if iszero(chord.applied)
            chord_key.scale
        else
            "major"  # ???
        end
    end
    scale = MODES[scale_name]

    chord_type = (chord.type > max_chord_type) ? max_chord_type : chord.type 
    applied_root = (applied_tonic + scale[chord_applied]) % 12
    chord_order = (iszero(chord.applied)) ? chord.root : chord_applied
    
    string(PITCH_KEY[applied_root], CHORDS[chord_type][scale_name][chord_order])
end


function read_json_data(jsonData)
    # parse json string
    song_data = JSON3.read(jsonData)

    # read keys
    #  the keys are stored in an array, imply that the pieces may contain different keys for these bars
    #  a key has attributes 
    #  {
    #    "beat": first beat of scale
    #    "scale": which mode
    #    "tonic": center note
    #  }
    song_keys = song_data.keys
    if length(song_keys) > 1
        # we want our snippets to be monotonic
        return
    end

    song_meters = song_data.meters

    # read chords and notes unit duration
    chords = song_data.chords
    notes = song_data.notes

    # some songs do not contain these informatino
    isempty(notes) && return
    isempty(chords) && return

    minimal_note_duration = minimum(notes) do note
        note.duration
    end
    minimal_chord_duration = minimum(chords) do chord
        chord.duration
    end

    minimal_duration = min(minimal_note_duration, minimal_chord_duration)
    
    # if the duration cannot divide one beat, we discard such song
    !iszero(1 % minimal_duration) && return
    (minimal_duration < 0.25) && return

    # create symbol vector
    note_token_vec = []
    chord_token_vec = []

    for note in notes
        # get MIDI pitch num
        pitch = parse_pitch(note, song_keys)

        # check duration and set resting token
        now_beats = length(note_token_vec) * minimal_duration + 1
        if note.beat > now_beats
            # rest detected
            rest_duration = (note.beat - now_beats) ÷ minimal_duration

            push!(note_token_vec, REST_TOKEN)
            for _ in 2:rest_duration
                push!(note_token_vec, CONTINUOUS_TOKEN)
            end
        end

        # record that note
        push!(note_token_vec, pitch)

        # neglect tuplets and swing rhythm 
        if note.duration % minimal_duration != 0
            return
        end

        note_dur = note.duration ÷ minimal_duration
        for _ in 2:note_dur
            push!(note_token_vec, CONTINUOUS_TOKEN)
        end
    end

    # process chords
    for chord in chords
        # get chord code
        chord_code = parse_chord(chord, song_keys)

        now_beats = length(chord_token_vec) * minimal_duration + 1
        if chord.beat > now_beats
            # rests
            rest_duration = (chord.beat - now_beats) ÷ minimal_duration

            push!(chord_token_vec, CHORD_REST_TOKEN)
            for _ in 2:rest_duration
                push!(chord_token_vec, CHORD_CONTINUOUS_TOKEN)
            end
        end

        # record that code
        push!(chord_token_vec, chord_code)
        chord_dur = chord.duration ÷ minimal_duration
        for _ in 2:chord_dur
            push!(chord_token_vec, CHORD_CONTINUOUS_TOKEN)
        end
    end

    (note_token_vec, minimal_duration, song_keys[1].tonic, chord_token_vec)
end

"""
    Read notes and chords information from xml string.

Now we donnot know what the `borrowed` integers mean, so just extract the notes information.
"""
function read_xml_data(xmlData)
    xmldata = try
        EzXML.parsexml(xmlData)
    catch
        return
    end
    # the following names follow the xml labels
    theorytab = root(xmldata)
    data = findfirst("//data", theorytab)

    # some different version of xml, not implemented
    if isnothing(data)
        return
    end

    notes = []
    chords = []

    for segment in EzXML.eachelement(data)
        melody = findfirst("//melody", segment)
        harmony = findfirst("//harmony", segment)

        # handle melody
        for voice in EzXML.eachelement(melody), note in EzXML.eachelement(findfirst("//notes", voice))
            # start_beat_abs = nodecontent(findfirst("start_beat_abs", note))
            note_length = nodecontent(findfirst("note_length", note))
            octave = nodecontent(findfirst("octave", note))
            scale_degree = nodecontent(findfirst("scale_degree", note))

            push!(notes, (note_length, octave, scale_degree))
        end

        # handle harmony
        # for chord in EzXML.eachelement(harmony)
        #     sd = nodecontent(findfirst("sd", chord))
        #     borrowed = nodecontent(findfirst("sd", chord))
        #     chord_durtaion = nodecontent(findfirst("chord_duration", chord))
            
        #     push!(chords, (sd, borrowed, chord_durtaion))
        # end
    end

    # The rests are in the notes, and thus can be easily encode
    encoded_notes = []
    song_key = KEYS[nodecontent(findfirst("//meta/key", theorytab))]
    song_mode_num = parse(Int, nodecontent(findfirst("//meta/mode", theorytab)))
    song_mode = MODES[song_mode_num]
    
    if isnothing(notes) || isempty(notes)
        return
    end

    minimal_dur = minimum(notes) do note
        parse(Float32, note[1])
    end

    for (len, octave, degree) in notes
        if degree == "rest"
            push!(encoded_notes, REST_TOKEN)
        else
            ss = findall('s', degree) |> length
            fs = findall('f', degree) |> length
            degree = replace(degree, "s" => "", "f" => "")

            note_pitch = CENTER_C + parse(Int, octave)*8 + song_key + song_mode[parse(Int, degree)]
            note_pitch += ss - fs

            push!(encoded_notes, note_pitch)
        end

        for _ in 2:(parse(Float32, len) ÷ minimal_dur)
            push!(encoded_notes, CONTINUOUS_TOKEN)
        end
    end

    (encoded_notes, minimal_dur, song_key, nothing)
end

function read_song_data(filepath; chords=false)
    # println("read file at ", filepath)

    # this json data is from GET request
    json_string = Base.read(filepath, String)
    json_data = JSON3.read(json_string)

    # check song data format
    if !isnothing(json_data.jsonData) && !isempty(json_data.jsonData)
        read_json_data(json_data.jsonData)
    elseif !isnothing(json_data.xmlData) && !isempty(json_data.xmlData) && !chords
        # since the chord extraction has som problem, when we need dataset with chords, skip it
        read_xml_data(json_data.xmlData)
    end
end
