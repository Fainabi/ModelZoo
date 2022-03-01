using Flux: onecold, onehot, softmax
using StatsBase
import PyCall

"""
    Generate melody with given model and optional seed. Seed needs integer representation.
"""
function generate_melody(model, token_map, seed=[]; seq_len, num_steps)

    # fill splitting notes into a vector with length `seq_len`
    start_symbols = fill(SPLITTING_TOKEN, seq_len)
    result = copy(start_symbols)

    start_symbols = [onehot(note, token_map) for note in start_symbols]  # one hot representation

    generated_notes = start_symbols

    for note in seed
        push!(result, note)
        note = onehot(note, token_map)

        push!(generated_notes, note)
    end


    # start generating
    for _ in 1:num_steps
        # since our model is trained with the seq_len parameter,
        # here we choose to push the sequence with that length into the model
        # to do prediction

        input = generated_notes[end-seq_len+1:end]

        probs = softmax(model(input))

        out_note = sample(token_map, Weights(probs))

        # record the generated notes
        push!(result, out_note)

        push!(generated_notes, onehot(out_note, token_map))

        # check melody endings
        if out_note == SPLITTING_TOKEN
            break
        end
    end

    result
end


function seq_to_midi(notes, step_duration; filename="out.midi")
    # replace splitting token to rest token
    notes = replace(notes, SPLITTING_TOKEN => REST_TOKEN)
    push!(notes, SPLITTING_TOKEN)

    music21 = PyCall.pyimport("music21")

    # music21 stream
    stream = music21.stream.Stream()

    # a simple finite state machine 
    start_symbol = nothing
    step_counter = 1
    for note in notes
        if note != CONTINUOUS_TOKEN
            if !isnothing(start_symbol) || note == SPLITTING_TOKEN
                # write it down
                quarter_length_duration = step_duration * step_counter

                event = if start_symbol == REST_TOKEN
                    # a rest note
                    music21.note.Rest(quarterLength=quarter_length_duration)
                else
                    # a regular pitched note
                    music21.note.Note(start_symbol, quarterLength=quarter_length_duration)
                end

                stream.append(event)  # a python style list appending by calling the method
            end

            start_symbol = note
            step_counter = 1
        else
            step_counter += 1
        end
    end

    stream.write("midi", filename)
end
